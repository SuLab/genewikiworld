import functools
from tqdm import tqdm
import xml.etree.ElementTree as ET

from get_counts import determine_node_type_and_get_counts, determine_p
from wikidataintegrator.wdi_core import WDItemEngine
from wikidataintegrator.wdi_config import config

# don't retry failed sparql queries. let them time out, and we'll skip
config['BACKOFF_MAX_TRIES'] = 1

execute_sparql_query = WDItemEngine.execute_sparql_query
# comment this ---v out to use official wikidata endpoint
execute_sparql_query = functools.partial(execute_sparql_query,
                                         endpoint="http://avalanche.scripps.edu:9999/bigdata/sparql")


def read_graphml(filename):
    tree = ET.parse(filename)
    return tree


def get_graph(root):
    for child in root:
        if child.tag.endswith('graph'):
            return child


def get_nodes(graph):
    nodes = []
    for child in graph.getchildren():
        if child.tag.endswith('node'):
            nodes.append(child)
    return nodes


def get_edges(graph):
    edges = []
    for child in graph.getchildren():
        if child.tag.endswith('edge'):
            edges.append(child)
    return edges


def get_node_edge_attrib_mappers(root):
    """
    Reads the graphml root and gets the id to attribute maps

    :param root: ElementTree.root
    :return: n_id_to_attrib, e_id_to_attrib, attribute ID to name mapper dicts
    """
    n_id_to_attrib = dict()
    e_id_to_attrib = dict()
    for child in root:
        # Desired attributes either have names, or 'yfiles.type' (typically graphical elements from yEd)
        attrib = child.attrib.get('attr.name', child.attrib.get('yfiles.type'))
        if attrib and child.attrib.get('for') == 'node':
            n_id_to_attrib[child.attrib.get('id')] = attrib
        elif attrib and child.attrib.get('for') == 'edge':
            e_id_to_attrib[child.attrib.get('id')] = attrib
    return n_id_to_attrib, e_id_to_attrib


def get_node_id_to_qid(nodes, n_id_to_attrib):
    node_id_to_qid = dict()
    for node in nodes:
        node_info = get_node_info(node, n_id_to_attrib)
        if node_info:
            node_id_to_qid[node.attrib.get('id')] = node_info[0]
    return node_id_to_qid


def get_node_info(node, n_id_to_attrib, collect_nodes=('NodeLabel', 'count', 'URL')):
    node_info = dict()
    props = dict()
    for child in node.getchildren():
        prop = n_id_to_attrib.get(child.attrib.get('key'), None)
        if prop in collect_nodes:
            # Collect numeric props as integers
            try:
                node_info[prop] = int(child.text)
            except (ValueError, TypeError):
                node_info[prop] = child.text
        elif type(prop) == str and prop.startswith('P'):
            # Wikidata Props (e.g. P31) will have counts... ensure you can cast to int...
            try:
                props[prop] = int(child.text)
            except (ValueError, TypeError):
                pass
    # not all counted nodes have properties, so returning empty dict if none...
    node_info['props'] = props
    if node_info.get('URL'):
        qid = node_info['URL'].split('/')[-1]
        return qid, node_info


def get_node_info_to_update(nodes, n_id_to_attrib, collect_nodes=('NodeLabel', 'count', 'URL')):
    node_info_to_update = dict()
    node_id_to_qid = dict()
    for node in nodes:
        node_info = get_node_info(node, n_id_to_attrib, collect_nodes)
        if node_info:
            node_id_to_qid[node.attrib.get('id')] = node_info[0]
            # Only need to update info on nodes with counts
            if 'count' in node_info[1]:
                node_info_to_update[node_info[0]] = node_info[1]
    return node_info_to_update


def get_edge_info_to_update(edges, node_id_to_qid, e_id_to_attrib):
    edge_info_to_update = dict()
    for edge in edges:
        s = node_id_to_qid.get(edge.attrib.get('source'))
        o = node_id_to_qid.get(edge.attrib.get('target'))

        # Some nodes may have been removed from the map, so no need to count them...
        if not s or not o:
            continue

        p = ''
        count = ''
        for c in edge.getchildren():
            if e_id_to_attrib.get(c.attrib.get('key')) == 'pid':
                p = c.text
            if e_id_to_attrib.get(c.attrib.get('key')) == 'count':
                count = int(c.text)
        edge_info_to_update[(s, p, o)] = count
    return edge_info_to_update


def count_prop(qid, prop, is_subclass, expand):
    p = determine_p(is_subclass, expand)
    q_string = """
    SELECT (count(*) as ?count) WHERE {
        ?item {p} wd:{qid} .
        ?item wdt:{prop} [] . }  
    """.replace('{p}', p).replace('{qid}', qid).replace('{prop}', prop)
    d = execute_sparql_query(q_string)['results']['bindings']
    return [int(x['count']['value']) for x in d][0]


def count_edges(s, p, o, s_subclass, s_expand, o_subclass, o_expand):
    p_sub = determine_p(s_subclass, s_expand)
    p_obj = determine_p(o_subclass, o_expand)
    q_string = """
    SELECT (count(distinct *) as ?count) WHERE {
        ?subject {p_sub} wd:{s} .
        ?subject wdt:{p} ?object .
        ?object {p_obj} wd:{o} }
    """.replace('{p_sub}', p_sub).replace('{s}', s).replace('{p}', p).replace('{p_obj}', p_obj).replace('{o}', o)
    d = execute_sparql_query(q_string)['results']['bindings']
    return [int(x['count']['value']) for x in d][0]


def update_node_counts(node_info_to_update, return_type_info=False):
    """
    Updates the counts for the nodes of the graphs. Data structure for this update is a little weird....

    :param node_info_to_update: dict, key = QID of node, val = output of get_node_info()
        structure of val:  {'NodeLabel': - str name of node,
                            'count': - int counts for the node,
                            'URL': str, url on WikiData for node,
                            'props': dict - {key = 'PID for WikiData Prop', val = int, counts}
                            }
    :param return_type_info: boolean, return the subclass_dict and expand_dict if True, in addition to update node_info

    :return: Dict, data of the same structure as node_info_to_update, with updated counts.
    """

    node_name_mapper = {k: v['NodeLabel'] for k, v in node_info_to_update.items()}
    new_counts, subclass, expand = determine_node_type_and_get_counts(node_info_to_update.keys(),
                                                                      node_name_mapper)
    node_info_updated = node_info_to_update.copy()
    for qid, new_count in new_counts.items():
        node_info_updated[qid]['count'] = new_count

    for qid, node_info in node_info_updated.items():
        updated_props = dict()
        for prop, count in tqdm(node_info['props'].items(), desc=node_name_mapper[qid]):
            updated_props[prop] = count_prop(qid, prop, subclass[qid], expand[qid])
        node_info_updated[qid]['props'] = updated_props

    # Sometimes we'll need the subclass and expand info for other functions and don't want to have to re-run
    if return_type_info:
        return node_info_updated, subclass, expand
    return node_info_updated


def update_edge_counts(edge_info_to_update, subclass=dict(), expand=dict()):
    """
    Updates the counts for the edges

    :param edge_info_to_update: list of dicts containing {'s', 'p', 'o', and 'c'} keys where:
        's' - subject qid
        'p' - predicate pid
        'o' - object qid
        'c' - int counts for spo triple
    :param subclass: dict, qid -> bool, wheather members of the qid is a 'sublcass of' (True) or 'instance of' (False)
        the parent
    :param expand: dict, qid -> bool, weather to expand down 'subclass of' links... e.g. wdt:P31/wdt:P279* for 'instance
        of' or wdt:P279* for 'subclass of'. If false, will limit to direct 'instance of' or 'subclass of' links

    :return: same data structure as input, with updated 'c' values...
    """
    updated_edge_info = dict()
    for edge_key, counts in tqdm(edge_info_to_update.items()):
        s = edge_key[0]
        p = edge_key[1]
        o = edge_key[2]

        # Default is 'instance_of' and to not expand /wdt:P279*....
        new_counts = count_edges(s, p, o, s_subclass=subclass.get(s, False), s_expand=expand.get(s, False),
                                 o_subclass=subclass.get(o, False), o_expand=expand.get(o, False))
        updated_edge_info[(s, p, o)] = new_counts

    return updated_edge_info


def parse_prop_text(prop_text):
    ### TODO: Some sort of contingincey for if theres two of the same count.....
    count_names = dict()
    for identifier in prop_text.split('\n'):
        line_split = identifier.split(': ')
        name = line_split[0]
        count = int(line_split[1].replace(',', ''))
        count_names[count] = name
    return count_names


def determine_prop_names(node, n_id_map):
    # Get info needed to update label
    props = dict()
    prop_text = None
    for child in node.getchildren():
        prop = n_id_map.get(child.attrib.get('key'))

        if type(prop) != str:
            continue

        if prop == 'node_prop_text':
            prop_text = child.text
        elif prop.startswith('P'):
            try:
                props[prop] = int(child.text)
            except ValueError:
                pass

    # Return nothing if no prop_text...
    if not prop_text:
        return dict()

    count_names = parse_prop_text(prop_text)

    prop_to_name = {k: count_names[v] for k, v in props.items() if v in count_names}
    return prop_to_name


def format_prop_counts_to_text(prop_counts):
    prop_text = []
    for k, v in sorted(prop_counts.items(), key=lambda x: x[1], reverse=True):
        prop_text.append(k + ': ' + '{:,}'.format(v))

    return '\n'.join(prop_text)


def update_node_data(node, node_info, n_to_qid, n_id_map, prop_names):
    """
    Warning, updates will be made inplace
    """

    # Update the counts
    nid = node.attrib.get('id')
    qid = n_to_qid.get(nid, None)

    # Some nodes don't have any new info to update... so skip
    if qid not in node_info or qid is None:
        return None

    for child in node.getchildren():
        prop = n_id_map.get(child.attrib.get('key'))

        if type(prop) != str:
            continue

        # Update Aquired Data
        if prop in node_info[qid]:
            child.text = str(node_info[qid][prop])
        elif prop in node_info[qid]['props']:
            child.text = str(node_info[qid]['props'][prop])

        # Special Labeling Props that need updating
        elif prop == 'labelcount':
            child.text = node_info[qid]['NodeLabel'] + '\n' + '{:,}'.format(node_info[qid]['count'])
        elif prop == 'node_prop_text':
            prop_text = []
            for k, v in sorted(node_info[qid]['props'].items(), key=lambda x: x[1], reverse=True):
                if k in prop_names:
                    prop_text.append(prop_names[k] + ': ' + '{:,}'.format(v))

            node_props = {prop_names[k]: v for k, v in node_info[qid]['props'].items() if k in prop_names}
            child.text = format_prop_counts_to_text(node_props)


def select_child(item, mapper, text):
    for child in item.getchildren():
        if mapper.get(child.attrib.get('key')) == text:
            return child


def update_graphics_labels_from_node_data(node, n_id_map, add_new_props=False):
    """Updates the graphics labels so they match the node-data"""

    gfx = select_child(node, n_id_map, 'nodegraphics').getchildren()[0].getchildren()
    node_label = select_child(node, n_id_map, 'labelcount').text
    node_props = select_child(node, n_id_map, 'node_prop_text').text

    i = 0
    for elem in gfx:
        if elem.tag.endswith('NodeLabel'):
            if i == 0:
                elem.text = node_label
                i += 1
            # not all nodes have a props-label
            elif i == 1 and node_props:
                # Add all properties to the label text, even if new
                if add_new_props:
                    elem.text = node_props
                # Otherwise only update the counts of those already there...
                elif not elem.text.strip():
                    # Or not there in this case...
                    continue
                else:
                    elem_prop_counts = parse_prop_text(elem.text)
                    node_prop_counts = parse_prop_text(node_props)
                    # Filter out previously removed properties
                    node_prop_counts = {v: k for k, v in node_prop_counts.items() if v in elem_prop_counts.values()}

                    # Update with the new counts
                    elem.text = format_prop_counts_to_text(node_prop_counts)


def get_key(edge, n_to_qid, e_id_map):
    s = n_to_qid.get(edge.attrib.get('source'))
    o = n_to_qid.get(edge.attrib.get('target'))
    p = select_child(edge, e_id_map, 'pid').text

    if s is None or p is None or o is None:
        return None

    return (s, p, o)


def update_edge_data(edge, edge_info, e_id_map, n_to_qid):
    edge_key = get_key(edge, n_to_qid, e_id_map)

    # Check to see if key is in updated edeges, otherwise, nothing to update
    if edge_key not in edge_info:
        return None

    count = select_child(edge, e_id_map, 'count')
    # Sometimes no data in the edge either...
    if count is None:
        return None
    count.text = str(edge_info[edge_key])

    label = select_child(edge, e_id_map, 'labelcount')
    # Ensure there's data in teh edge
    if label is None:
        return None
    label.text = select_child(edge, e_id_map, 'label').text + ' ({:,})'.format(edge_info[edge_key])


def update_edge_graphics_label(edge, e_id_map):
    gfx = select_child(edge, e_id_map, 'edgegraphics').getchildren()[0].getchildren()
    edge_label = select_child(edge, e_id_map, 'labelcount').text

    # Some edges have no label, so skip
    if not edge_label:
        return None

    for elem in gfx:
        if elem.tag.endswith('EdgeLabel'):
            if elem.text is None:
                return None
            # No parantheses means no counts, so don't update.... May have been manually adjusted
            elif '(' not in elem.text or ')' not in elem.text:
                return None
            else:
                elem.text = edge_label
