import os
import sys
import argparse
import functools
from tqdm import tqdm
from copy import deepcopy
from itertools import chain
import xml.etree.ElementTree as ET
import get_counts as gc


def change_endpoint(endpoint):
    gc.change_endpoint(endpoint)


def get_namespaces(filename):
    from lxml import etree
    root = etree.parse(filename).getroot()
    return list(root.nsmap.items())

def read_graphml(filename):

    ns_map = get_namespaces(filename)
    for prefix, ns in ns_map:
        if prefix is None:
            ET.register_namespace('', ns)
        else:
            ET.register_namespace(prefix, ns)
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


def create_new_graph_property(wd_prop_id, new_id):
    """Creates a new XML tag for a property that can be inserted into the proper locaiton"""
    prop = ET.Element('key')
    prop.attrib['attr.name'] = wd_prop_id
    prop.attrib['attr.type'] = "int"
    prop.attrib['for'] = "node"
    prop.attrib['id'] = 'd'+str(new_id)

    return prop

def get_max_prop(root):
    """Returns the number of the mx property in the current GraphML file"""
    return max(int(c.get('id', 'd000')[1:]) for c in root if c.get('id') != 'G')


def insert_prop(prop, root):
    """Insert new property into the root graphml"""
    # insert at the correct line number
    insert_line_num = int(prop.attrib.get('id')[1:])
    # Ensure newline and proper indent level
    prop.tail = root[insert_line_num - 1].tail
    root.insert(insert_line_num, prop)


def is_prop_id(prop_id):
    """Prop IDs: P123, etc..."""
    # Ensure After P is an integer...
    try:
       int(prop_id[1:])
    except (ValueError, TypeError):
        return False
    # Ensure starts with P
    return prop_id.startswith('P')


def determine_new_props_for_graph(root, node_info_updated):
    """Finds the properties that are new to the graphml"""
    all_orig_props = set(c.attrib['attr.name'] for c in root if is_prop_id(c.attrib.get('attr.name')))
    all_new_props = set(list(chain(*[n.get('props', dict()).keys() for n in node_info_updated.values()])))

    return all_new_props - all_orig_props

def get_node_id_to_qid(nodes, n_id_to_attrib):
    node_id_to_qid = dict()
    for node in nodes:
        node_info = get_node_info(node, n_id_to_attrib)
        if node_info:
            node_id_to_qid[node.attrib.get('id')] = node_info[0]
    return node_id_to_qid


def get_node_info(node, n_id_to_attrib, prop_names=None, collect_nodes=('NodeLabel', 'count', 'URL', 'node_prop_text')):
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
    # Filter the props to only the ones aleady dispalyed in the node label text
    if prop_names is not None:
        props = {p: c for p, c in props.items() if prop_names.get(p, '') in node_info['node_prop_text']}
    # not all counted nodes have properties, so returning empty dict if none...
    node_info.pop('node_prop_text')
    node_info['props'] = props
    if node_info.get('URL'):
        qid = node_info['URL'].split('/')[-1]
        return qid, node_info


def get_node_info_to_update(nodes, n_id_to_attrib, prop_names, collect_nodes=('NodeLabel', 'count', 'URL', 'node_prop_text')):
    node_info_to_update = dict()
    node_id_to_qid = dict()
    for node in nodes:
        node_info = get_node_info(node, n_id_to_attrib, prop_names, collect_nodes)
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
    p = gc.determine_p(is_subclass, expand)
    q_string = """
    SELECT (count(?item) as ?count) where {
        SELECT DISTINCT ?item WHERE {
            ?item {p} wd:{qid} .
            ?item wdt:{prop} [] . }}
    """.replace('{p}', p).replace('{qid}', qid).replace('{prop}', prop)
    print("A1: "+q_string)
    try:
        d = gc.execute_sparql_query(q_string)['results']['bindings']
        print("A2: "+str(d))
        prop_count = [int(x['count']['value']) for x in d][0]
    except:
        prop_count = -1
    print("A3: "+str(prop_count))
    return prop_count


def count_edges(s, p, o, s_subclass, s_expand, o_subclass, o_expand):
    p_sub = gc.determine_p(s_subclass, s_expand)
    p_obj = gc.determine_p(o_subclass, o_expand)

    # test for reciprocal relationships that need to be collapsed
    recip_rels = {'P527': 'P361',
                      'P361': 'P527',
                      'P2176': 'P2175',
                      'P2175': 'P2176',
                      'P702': 'P688',
                      'P688': 'P702',
                      'P1343': 'P4510',
                      'P4510': 'P1343',
                      'P828': 'P1542',
                      'P1542': 'P828',
                      'P3781': 'P3780',
                      'P3780': 'P3781'}

    if p in recip_rels.keys():
       u =  """UNION
               {?object wdt:"""+recip_rels[p]+""" ?subject .}"""
    else:
       u = ""

    q_string = """
    SELECT (count(distinct *) as ?count) WHERE {
        ?subject {p_sub} wd:{s} .
        {
            {?subject wdt:{p} ?object .}
            {u}
        }
        ?object {p_obj} wd:{o} }
    """.replace('{p_sub}', p_sub).replace('{s}', s).replace('{p}', p).replace('{p_obj}', p_obj).replace('{o}', o).replace('{u}',u)
    print("B1: "+q_string)
    try :
        d = gc.execute_sparql_query(q_string)['results']['bindings']
        print("B2: "+str(d))
        edge_count = [int(x['count']['value']) for x in d][0]
    except:
        edge_count = -1
    print("B3: "+str(edge_count))
    return edge_count


def update_node_props(node_info_updated, min_counts, filt_props):
    """ Updates all current properties and counts for nodes"""
    for qid, node_info in node_info_updated.items():
        # Query for the properties and counts
        node_external_ids = gc.get_external_ids(qid)

        # Format results into proper data structure
        prop_results = dict()
        for pid, label, prop_count in node_external_ids:
            prop_results[pid] = prop_count

        # Filter the props
        prop_count_thresh = max(node_info_updated[qid]['count']*filt_props,min_counts)
        node_info_updated[qid]['props'] = {k: v for k, v in prop_results.items() if v > prop_count_thresh}

    return node_info_updated


def update_prop_counts(node_info_updated, node_name_mapper, subclass, expand):
    for qid, node_info in node_info_updated.items():
        updated_props = dict()
        for prop, count in tqdm(node_info['props'].items(), desc=node_name_mapper[qid]):
            updated_props[prop] = count_prop(qid, prop, subclass[qid], expand[qid])
        node_info_updated[qid]['props'] = updated_props
    return node_info_updated


def update_node_properties_and_counts(node_info_to_update, return_type_info=False, add_new_props=False,
                                      min_counts=200, filt_props=0.05):
    """
    Updates the counts for the nodes of the graphs *and the property list. Data structure for this
    update is a little weird....

    :param node_info_to_update: dict, key = QID of node, val = output of get_node_info()
        structure of val:  {'NodeLabel': - str name of node,
                            'count': - int counts for the node,
                            'URL': str, url on WikiData for node,
                            'props': dict - {key = 'PID for WikiData Prop', val = int, counts}
                            }
    :param return_type_info: boolean, return the subclass_dict and expand_dict if True, in addition to update node_info
    :param add_new_pros: boolean, set True, will return all current props and counts for the node, filtered
        according to min_counts, and filt_props. If False, will only update counts for properties already
        existing within the .graphml file, without querying for new ones.'
    :param min_counts: int, the minimum number of counts a property needs to be included. Only used if add_new_props is
        True.
    :param filt_props: float, fraction of nodes a prop must be present on to be included. Only used if add_new_pros is
        True.

    :return: dict, data of the same structure as node_info_to_update, with updated counts.
    """
    node_info_updated = deepcopy(node_info_to_update)
    node_name_mapper = {k: v['NodeLabel'] for k, v in node_info_to_update.items()}
    new_counts, subclass, expand = gc.determine_node_type_and_get_counts(node_info_to_update.keys(),
                                                                         node_name_mapper)
    for qid, new_count in new_counts.items():
        node_info_updated[qid]['count'] = new_count

    # Update the properties, either getting a revised count, or searching for new properties and getting counts.
    if add_new_props:
        node_info_updated = update_node_props(node_info_updated, min_counts, filt_props)
    else:
        node_info_updated = update_prop_counts(node_info_updated, node_name_mapper, subclass, expand)

    # Sometimes we'll need the subclass and expand info for other functions and don't want to have to re-run
    if return_type_info:
        return node_info_updated, subclass, expand
    return node_info_updated


def update_edge_counts(edge_info_to_update, subclass=dict(), expand=dict()):
    """
    Updates the counts for the edges

    :param edge_info_to_update: dict with key = tuple (s, p, o) and val = int counts of the edge:
        s - subject qid
        p - predicate pid
        o - object qid
    :param subclass: dict, qid -> bool, wheather members of the qid is a 'sublcass of' (True) or 'instance of' (False)
        the parent
    :param expand: dict, qid -> bool, weather to expand down 'subclass of' links... e.g. wdt:P31/wdt:P279* for 'instance
        of' or wdt:P279* for 'subclass of'. If false, will limit to direct 'instance of' or 'subclass of' links

    :return: same data structure as input, with updated count values...
    """
    updated_edge_info = dict()
    for edge_key, counts in tqdm(edge_info_to_update.items()):
        s = edge_key[0]
        p = edge_key[1]
        o = edge_key[2]

        if s is None or p is None or o is None:
            updated_edge_info[(s, p, o)] = None
            continue

        # Default is 'instance_of' and to not expand /wdt:P279*....
        new_counts = count_edges(s, p, o, s_subclass=subclass.get(s, False), s_expand=expand.get(s, False),
                                 o_subclass=subclass.get(o, False), o_expand=expand.get(o, False))
        updated_edge_info[(s, p, o)] = new_counts

    return updated_edge_info


def format_prop_counts_to_text(prop_counts):
    prop_text = []
    for k, v in sorted(prop_counts.items(), key=lambda x: x[1], reverse=True):
        prop_text.append(k + ': ' + '{:,}'.format(v))

    return '\n'.join(prop_text)


def determine_new_props_for_single_node(node, node_info, n_id_map):
    """ Figure out which props in a node are new and will need to be inserted"""
    new_props = set(node_info.get('props', dict()).keys())
    old_props = set(n_id_map.get(prop.attrib.get('key', ''), None) for prop in node)
    old_props = set(p for p in old_props if is_prop_id(p))

    return new_props - old_props


def create_node_property(graph_key, count, tail):
    """Creates a new XML tag for a property that can be inserted into the proper locaiton"""
    prop = ET.Element('data')
    prop.attrib['key'] = graph_key
    prop.attrib['xml:space'] = "preserve"
    prop.text = str(count)
    prop.tail = tail

    return prop

def update_node_data(node, node_info, n_to_qid, n_id_map, reverse_nid_map, prop_names):
    """
    Warning, updates will be made inplace
    """

    # Update the counts
    nid = node.attrib.get('id')
    qid = n_to_qid.get(nid, None)

    # Some nodes don't have any new info to update... so skip
    if qid not in node_info or qid is None:
        return None

    # Find properties that haven't been found before
    new_props = determine_new_props_for_single_node(node, node_info[qid], n_id_map)
    to_remove = []

    # Start with in-place update for props that are not new
    for child in node.getchildren():
        prop = n_id_map.get(child.attrib.get('key'))

        if type(prop) != str:
            continue

        # Update Aquired Data
        if prop in node_info[qid]:
            child.text = str(node_info[qid][prop])
        elif prop in node_info[qid]['props']:
            child.text = str(node_info[qid]['props'][prop])
        elif is_prop_id(prop):
            to_remove.append(child)

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

    # remove properties that are no longer represented on this node
    for r in to_remove:
        node.remove(r)

    # Some prerequsities before adding new properites
    if new_props:
        max_prop = 0
        # Get the largest index of a property (so we can insert directly after)
        for i, p in enumerate(node):
            if is_prop_id(n_id_map.get(p.attrib.get('key', ''))):
                max_prop = i

        # Save the tail for the tag
        tail = node[max_prop].tail

    # insert the new properties
    for prop in new_props:
        max_prop += 1
        graph_key = reverse_nid_map[prop]
        count = node_info[qid]['props'][prop]
        new_prop = create_node_property(graph_key, count, tail)
        node.insert(max_prop, new_prop)


def select_child(item, mapper, text):
    for child in item.getchildren():
        if mapper.get(child.attrib.get('key')) == text:
            return child


def update_graphics_labels_from_node_data(node, n_id_map, add_new_props):
    """Updates the graphics labels so they match the node-data"""

    try:
        gfx = select_child(node, n_id_map, 'nodegraphics').getchildren()[0].getchildren()
    except:
        return None
    node_label = select_child(node, n_id_map, 'labelcount').text
    node_props = select_child(node, n_id_map, 'node_prop_text').text

    # Nodes have either 0, 1, or 2 node labels. If 1, its just title and count
    # If 2, the first one is title count, second is properties and counts
    i = 0
    for elem in gfx:
        if elem.tag.endswith('NodeLabel'):
            if i == 0:
                elem.text = node_label
                i += 1
            # not all nodes have a props-label
            elif i == 1 and node_props:
                # Add all properties to the label text, even if new
                elem.text = node_props


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
    # get the edge grapthics
    try:
        gfx = select_child(edge, e_id_map, 'edgegraphics').getchildren()[0].getchildren()
    # No graphics, then don't do anything
    except:
        return None
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


def update_graphml_file(filename, outname=None, add_new_props=False, min_counts=200, filt_props=0.05, endpoint=None):

    # Use the desired endpoint
    if endpoint is not None:
        gc.change_endpoint(endpoint)

    # Read the graphml file and break into nodes and edged
    tree = read_graphml(filename)
    root = tree.getroot()
    graph = get_graph(root)
    nodes = get_nodes(graph)
    edges = get_edges(graph)

    prop_names = gc.get_prop_labels()

    # Get info specific to this WikiData items from the graph
    n_id_map, e_id_map = get_node_edge_attrib_mappers(root)
    n_to_qid = get_node_id_to_qid(nodes, n_id_map)
    node_info = get_node_info_to_update(nodes, n_id_map, prop_names)
    edge_info = get_edge_info_to_update(edges, n_to_qid, e_id_map)

    # Query wikidata instance for updated count info
    new_node_info, sub, ext = update_node_properties_and_counts(node_info, True, add_new_props, min_counts, filt_props)
    new_edge_info = update_edge_counts(edge_info, sub, ext)

    # Add any new properites to the graphml
    if add_new_props:
        # Find the new properties
        new_props = determine_new_props_for_graph(root, new_node_info)
        if new_props:
            # Find the highest datakey in the graph
            max_prop = get_max_prop(root)+1
            for prop in new_props:
                # Create the new element and insert into the graphml
                new_prop = create_new_graph_property(prop, str(max_prop))
                insert_prop(new_prop, root)
                # Add the new property to the map
                n_id_map['d'+str(max_prop)] = prop
                max_prop += 1

    reverse_nid_map = {v: k for k, v in n_id_map.items()}

    # Apply new count data to nodes
    for node in nodes:
        update_node_data(node, new_node_info, n_to_qid, n_id_map, reverse_nid_map, prop_names)
        update_graphics_labels_from_node_data(node, n_id_map, add_new_props)

    # Aooly new count data to edges
    for edge in edges:
        update_edge_data(edge, new_edge_info, e_id_map, n_to_qid)
        update_edge_graphics_label(edge, e_id_map)

    # Write out the file
    if outname is None:
        outname = os.path.splitext(filename)
        outname = outname[0] + '_out' + outname[1]
    print('Writing file to:', outname)
    tree.write(outname, encoding='utf-8', xml_declaration=True)


if __name__ == '__main__':
    # Command line Parsing
    parser = argparse.ArgumentParser(description='Update the counts on a .graphml file from WikiData (or antoher Wikibase)')
    parser.add_argument('filename', help="The name of the graphml file that will be updated", type=str)
    parser.add_argument('-o', '--outname', help="The name of the output file", type=str, default=None)
    parser.add_argument('-p', '--add_new_props', help='Search for new x-refs on node info', action='store_true')
    parser.add_argument('-c', '--min_counts', help="The mininum nubmer of counts a new property must have"+
                            " to be included (defaults 200)", type=int, default=200)
    parser.add_argument('-f', '--filt_props', help="The fraction of the total number of counts for a node that a"+
                            " property must have to be included (default 0.05)", type=float, default=0.05)
    parser.add_argument('-e', '--endpoint', help='Use a wikibase endpoint other than standard wikidata', type=str,
                        default=None)

    # Unpack commandline arguments
    args = parser.parse_args()
    filename = args.filename
    outname = args.outname
    add_new_props = args.add_new_props
    min_counts = args.min_counts
    filt_props = args.filt_props
    endpoint = args.endpoint

    # run the routine
    update_graphml_file(filename, outname, add_new_props, min_counts, filt_props, endpoint)

