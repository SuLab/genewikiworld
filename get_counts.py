"""
Get counts for all items we care about, and some we don't ....
"""
import time
import functools
from copy import deepcopy
import numpy as np

import requests
from networkx.readwrite.graphml import write_graphml_xml as _write_graphml
from requests import HTTPError
from tqdm import tqdm
from wikidataintegrator.wdi_core import WDItemEngine
import networkx as nx
from matplotlib import pyplot as plt

from wikidataintegrator.wdi_config import config

# don't retry failed sparql queries. let them time out, and we'll skip
config['BACKOFF_MAX_TRIES'] = 1

execute_sparql_query = WDItemEngine.execute_sparql_query
# comment this ---v out to use official wikidata endpoint
#execute_sparql_query = functools.partial(execute_sparql_query,
#                                         endpoint="http://avalanche.scripps.edu:9999/bigdata/sparql")

# instance of subject, subclass of object
special_edges = [('Q11173', 'P1542', 'Q21167512'),  # chemical, cause of, chemical hazard
                 ('Q12136', 'P780', 'Q169872'),  # disease, symptom, symptom
                 ('Q12136', 'P780', 'Q1441305'),  # disease, symptom, medical sign
                 ('Q21167512', 'P780', 'Q169872')]  # chemical hazard, symptom, symptom

special_starts = [q[:2] for q in special_edges]


def change_endpoint(endpoint):
    global execute_sparql_query
    execute_sparql_query = functools.partial(execute_sparql_query, endpoint=endpoint)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def getConceptLabel(qid):
    return getConceptLabels((qid,))[qid]


def getConceptLabels(qids):
    out = dict()
    for chunk in chunks(list(set(qids)), 50):
        this_qids = {qid.replace("wd:", "") if qid.startswith("wd:") else qid for qid in chunk}
        # Found Some results that begin with 't' and cause request to return no results
        bad_ids = {qid for qid in this_qids if not qid.startswith('Q')}
        this_qids = '|'.join(this_qids - bad_ids)
        params = {'action': 'wbgetentities', 'ids': this_qids, 'languages': 'en', 'format': 'json', 'props': 'labels'}
        r = requests.get("https://www.wikidata.org/w/api.php", params=params)
        r.raise_for_status()
        wd = r.json()['entities']
        # Use empty labels for the bad ids
        wd.update({bad_id: {'labels': {'en': {'value': ""}}} for bad_id in bad_ids})
        out.update({k: v['labels'].get('en', dict()).get('value', '') for k, v in wd.items()})
    return out


def get_prop_labels():
    """ returns a dict of labels for all properties in wikidata """
    s = """
    SELECT DISTINCT ?property ?propertyLabel
    WHERE {
        ?property a wikibase:Property .
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }"""
    print("A1: "+s)
    try:
        d = execute_sparql_query(s)['results']['bindings']
#        print("A2: "+str(d))
    except:
        print("***** FAILED SPARQL *****")
        d = []
    d = {x['property']['value'].replace("http://www.wikidata.org/entity/", ""):
                x['propertyLabel']['value'] for x in d}
    return d


def determine_p(use_subclass, extend=True):
    p = "wdt:P279*" if use_subclass else "wdt:P31/wdt:P279*"
    # Option to not extend down 'subclass_of' edges (useful for highly populated node types)
    if not extend:
        p = p.replace('/wdt:P279*', '').replace('*', '')
    return p


def is_subclass(qid, return_val=False):
    instance = get_type_count(qid, use_subclass=False, extend_subclass=False)
    subclass = get_type_count(qid, use_subclass=True, extend_subclass=False)

    # If the numbers are close, we need to determine if its because some have both subclass and instance of values
    if instance != 0 and subclass != 0 and abs(np.log10(instance) - np.log10(subclass)) >= 1:

        p0 = "wdt:P31"
        p1 = "wdt:P279"

        s_both = """
        SELECT (COUNT(DISTINCT ?item) as ?c) WHERE {
          ?item {p0} {wds} .
          ?item {p1} {wds} .
        }
        """
        both = execute_sparql_query(s_both.replace("{wds}", "wd:" + qid)
                                    .replace("{p0}", p0).replace("{p1}", p1))['results']['bindings']
        both = {qid: int(x['c']['value']) for x in both}.popitem()[1]

        is_sub = subclass - both > instance
    else:
        is_sub = subclass > instance

    if return_val:
        return is_sub, subclass if is_sub else instance
    return is_sub


def get_type_count(qid, use_subclass=False, extend_subclass=True):
    """
    For each qid, get the number of items that are instance of (types) this qid
    """
    p = determine_p(use_subclass, extend_subclass)
    s = """
    SELECT (COUNT(DISTINCT ?item) as ?c) WHERE {
      ?item {p} {wds}
    }
    """.replace("{wds}", "wd:" + qid).replace("{p}", p)
    try:
        d = execute_sparql_query(s)['results']['bindings']
    except:
        d = []
        print("***** FAILED SPARQL *****")

    return {qid: int(x['c']['value']) for x in d}.popitem()[1]


def get_type_edge_frequency(type_qid, direction='out', use_subclass=False, extend_subclass=True):
    """
    gets properties on items that are an instance of type_qid. returns all kinds of props, (i.e. including
        externalid and wditem). Filtering by item type causes query timeouts
    if direction = "in": returns "incoming"/"reverse"/"what links here" edges, gets properties on items that are the
        subjects where the object is an instance of type_qid
    """
    assert direction in {'in', 'out'}
    p = determine_p(use_subclass, extend_subclass)
    subject = "subject"
    if direction == "in":
        subject = "object"

    ### AS temporarily add limit of 1000000 for debugging only!!!
    s = """SELECT ?property ?count WHERE {
              {
              SELECT ?property (COUNT(*) AS ?count) WHERE {
                    {SELECT DISTINCT ?subject ?property ?object WHERE {
                        ?{subject} {p} wd:{xxx} .
                        ?subject ?property ?object .}}
                 } GROUP BY ?property
              }
            } ORDER BY DESC (?count)""".replace("{xxx}", type_qid).replace("{p}", p).replace("{subject}", subject)
    print("B1: "+s)
    try:
        d = execute_sparql_query(s)['results']['bindings']
#        print("B2: "+str(d))
    except:
        print("***** FAILED SPARQL *****")
        d = []
    r = {x['property']['value'].split('/')[-1]:
             int(x['count']['value']) for x in d if 'http://www.wikidata.org/prop/direct' in x['property']['value']}
#    print("B3: "+str(r))
    return r


def get_external_ids(type_qid, use_subclass=False, extend_subclass=True):
    """
    Given the qid of a type, get all external id props for items of this type
    ranked by usage count
    :param type_qid: QID of item type
    """
    p = determine_p(use_subclass, extend_subclass)
    s = """
    SELECT ?property ?propertyLabel ?propertyDescription ?count WHERE {
      {
        select ?propertyclaim (COUNT(*) AS ?count) where {
          {select distinct ?item ?propertyclaim where {
            ?item {p} wd:{type_qid} .
            ?item ?propertyclaim [] . }}
        } group by ?propertyclaim
      }
      ?property wikibase:propertyType wikibase:ExternalId .
      ?property wikibase:claim ?propertyclaim .
      SERVICE wikibase:label {bd:serviceParam wikibase:language "en" .}
    } ORDER BY DESC (?count)""".replace("{type_qid}", type_qid).replace("{p}", p)

    # remove distinct requirement if just using 'instance of' and not
    if not extend_subclass:
        s = s.replace('      {select distinct ?item ?propertyclaim where {\n', '').replace('}}', '')

    print("Z1: "+s)
    try:
        d = execute_sparql_query(s)['results']['bindings']
    except:
        print("***** FAILED SPARQL *****")
        d = []
    return [(x['property']['value'].replace("http://www.wikidata.org/entity/", ""),
             x['propertyLabel']['value'],
             int(x['count']['value'])) for x in d]


def get_connecting_types(qid, pid, direction='out', use_subclass_subject=False, include_subclass_object=False,
                         extend_subclass=True):
    """
    Given a subject item type (by its qid, ex: Q12136 (disease))
    and a property (by its pid, ex: P2176 (drug used for treatment))
    Get me the types of objects it connects to, grouped by counts
    :param qid: subject items are of type "qid"
    :param pid: subject items are connected to object using this predicate's pid
    :param direction: {'in', 'out'} incoming or outgoing (default) edges
    :param use_subclass_subject: subject items are "subclasss of" type "qid", instead of "instance of"
    :param include_subclass_object: predicate items are "subclasss of" OR "instance of" type
    """
    p_sub = determine_p(use_subclass_subject, extend_subclass)

    left, right = 'subject', 'object'
    if direction == 'in':
        left, right = right, left
    p_obj = "wdt:P31|wdt:P279" if include_subclass_object else "wdt:P31"

    s = """select ?type (COUNT(*) AS ?count) where {{
      {{select distinct ?subject ?object ?type where {{
        ?subject {p_sub} wd:{qid} .
        ?{left} wdt:{pid} ?{right} .
        ?object {p_obj} ?type . }} }}
    }} group by ?type""".format(p_sub=p_sub, qid=qid, pid=pid, p_obj=p_obj, left=left, right=right)
    d = execute_sparql_query(s)['results']['bindings']
    return {x['type']['value'].replace("http://www.wikidata.org/entity/", ""): int(x['count']['value']) for x in d}


def determine_node_type_and_get_counts(node_ids, name_map=dict(), max_size_for_expansion=200000):
    # get all node counts for my special types
    subclass_nodes = dict()
    expand_nodes = dict()
    type_count = dict()

    # These nodes we've seeded and are all 'instance_of' or they are very large and should waste time expanding
    # Down subclasses.
    expand_nodes = {q: False for q in ['Q11173', 'Q2996394', 'Q14860489', 'Q5058355', 'Q13442814', 'Q16521']}

    time.sleep(0.5)  # Sometimes TQDM prints early, so sleep will endure messages are printed before TQDM starts
    t = tqdm(node_ids)
    for qid in t:
        t.set_description(name_map[qid])
        t.refresh()
        is_sub, count = is_subclass(qid, True)

        subclass_nodes[qid] = is_sub
        if qid not in expand_nodes:
            expand_nodes[qid] = count <= max_size_for_expansion
        if expand_nodes[qid]:
            # Small number of nodes, so expand the sublcass...
            count_ext = get_type_count(qid, use_subclass=is_sub, extend_subclass=True)
            # Ensure this is still ok (some will baloon like chemical Compound)
            expand_nodes[qid] = count_ext <= max_size_for_expansion
            # If its still ok, update the counts
            if expand_nodes[qid]:
                count = count_ext
        type_count[qid] = count
    return type_count, subclass_nodes, expand_nodes


def search_metagraph_from_seeds(seed_nodes, skip_types=('Q13442814', 'Q16521'), min_counts=200,
                                max_size_for_expansion=200000):
    # Make set for easy operations
    skip_types = set(skip_types)

    print("Getting type counts")
    time.sleep(0.5)  # Sometimes TQDM prints early, so sleep will endure messages are printed before TQDM starts
    type_count, subclass_nodes, expand_nodes = determine_node_type_and_get_counts(seed_nodes.keys(),
                                                                                  seed_nodes,
                                                                                  max_size_for_expansion)
    print("Getting type counts: Done")
    print("C1 type_count **** : "+str(type_count))
    print("C2 subclass_nodes: "+str(subclass_nodes))
    print("C3 expand_nodes: "+str(expand_nodes))

    # for each types of item, get the external ID props items of this type use
    print("Getting type external ids props")
    time.sleep(0.5)
    type_prop_id = dict()
    t = tqdm(subclass_nodes.items())
    for qid, is_sub in t:
        t.set_description(seed_nodes[qid])
        t.refresh()
        props = get_external_ids(qid, use_subclass=is_sub, extend_subclass=expand_nodes[qid])
        props = {x[0]: x[2] for x in props}  # id: count
        type_prop_id[qid] = props
    print("Getting type external ids props: Done")
    print("D1 type_prop_id ****: "+ str(type_prop_id))

    # for each types of item, get the WikibaseItem props it uses (using the seed nodes)
    print("Getting outgoing props for each type")
    time.sleep(0.5)
    type_props_out = dict()
    t = tqdm(set(seed_nodes.keys()) - skip_types)
    for qid in t:
        t.set_description(seed_nodes[qid])
        t.refresh()
        props = get_type_edge_frequency(qid, direction='out', use_subclass=subclass_nodes[qid],
                                        extend_subclass=expand_nodes[qid])
        # remove external id props
        props = {k: v for k, v in props.items() if k not in type_prop_id[qid]}
        type_props_out[qid] = props
    print("E1 type_props_out: "+ str(type_props_out))

    # and incoming
    print("Getting incoming props for each type")
    time.sleep(0.5)
    type_props_in = dict()
    t = tqdm(set(seed_nodes.keys()) - skip_types)
    for qid in t:
        t.set_description(seed_nodes[qid])
        t.refresh()
        props = get_type_edge_frequency(qid, direction='in', use_subclass=subclass_nodes[qid],
                                        extend_subclass=expand_nodes[qid])
        type_props_in[qid] = props
    print("Done")
    print("F1 type_props_in: "+ str(type_props_in))

    # get the types of items that each subject -> property edge connects to
    print("Getting type of objects each (subject, predicate) connects to")
    time.sleep(0.5)
    spo = dict()
    t = tqdm(type_props_out.items())
    for qid, props in t:
        t.set_description(seed_nodes[qid])
        t.refresh()
        spo[qid] = dict()
        props = {k: v for k, v in props.items() if v > min_counts}
        for pid, count in tqdm(props.items()):
            subclass_obj = (qid, pid) in [q[:2] for q in special_starts]
            conn_types = get_connecting_types(qid, pid, direction='out', use_subclass_subject=subclass_nodes[qid],
                                              include_subclass_object=subclass_obj, extend_subclass=expand_nodes[qid])
            spo[qid][pid] = conn_types
    print("G1 spo ****: "+str(spo))
    
    # and incoming
    print("Getting type of objects each (subject, predicate) connects from")
    time.sleep(0.5)
    spo_in = dict()
    t = tqdm(type_props_in.items())
    for qid, props in t:
        t.set_description(seed_nodes[qid])
        t.refresh()
        spo_in[qid] = dict()
        props = {k: v for k, v in props.items() if v > min_counts}
        for pid, count in tqdm(props.items()):
            subclass_obj = (qid, pid) in [q[:2] for q in special_starts]
            conn_types = get_connecting_types(qid, pid, direction='in', use_subclass_subject=subclass_nodes[qid],
                                              include_subclass_object=subclass_obj, extend_subclass=expand_nodes[qid])
            spo_in[qid][pid] = conn_types
    print("Done")
    print("H1 spo_in **** : "+str(spo_in))

    return type_count, type_prop_id, spo, spo_in


def remove_overlap(spo, spo_in, qid, p, p_in, filt_val=None):

    overlap = set(spo[qid][p]).intersection(set(spo_in[qid][p_in].keys()))
    for obj in overlap:
        out_c =  spo[qid][p][obj]
        in_c = spo_in[qid][p_in][obj]

        # Dont filter if it don't meet the 'closeness threshold'
        if filt_val:
            if not 1 >= out_c / in_c > filt_val or not 1 >= in_c / out_c > filt_val:
                continue

        # Remove the one with smaller counts...
        if in_c > out_c:
            spo[qid][p].pop(obj)
        else:
            spo_in[qid][p_in].pop(obj)

    return spo, spo_in


def remove_reciprocal_edges(spo, spo_in, recip_rels, filt_val=.95):

    filt_spo = deepcopy(spo)
    filt_spo_in = deepcopy(spo_in)

    for qid in spo:
        p_out = filt_spo[qid].keys()
        p_in = filt_spo_in[qid].keys()

        for p1 in p_out:
            # First remove reciprocal edges
            p2 = recip_rels.get(p1)
            if p2 and p2 in p_in:
                filt_spo, filt_spo_in = remove_overlap(filt_spo, filt_spo_in, qid, p1, p2, None)

            # Secondly  look for the same edge in and out and only keep the larger instance...
            if p1 in p_in:
                filt_spo, filt_spo_in = remove_overlap(filt_spo, filt_spo_in, qid, p1, p1, filt_val=filt_val)

    return filt_spo, filt_spo_in


def create_graph(type_count, type_prop_id, spo, spo_in, min_counts=200, filt_props=0.05, recip_rels=None):
#
# AS 2019-08-30: I think for a given node (object type), a property is only listed in the node_prop_text
#                node attribute if it is used more than _both_ min_counts and the number of items * filt_props
#

    ############
    # construct the network
    ############
    # new nx 2.X we need unique keys for each edge
    def genkeys(total=100000000000000000):
        for i in range(total):
            yield i

    keygen = genkeys()

    # White and blacklists for nodes and edges... edit if changes are needed
    banned_nodes = ['Q4167836', 'Q47461807', 'Q24017414', 'Q47461827']
    banned_edges = ['P921', 'P910', 'P301']
    whitelisted_nodes = ['Q30612', 'Q930752', 'Q7251477']

    # Reciprical relationships to replace with a single edge
    if recip_rels is None:
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

    prop_labels = get_prop_labels()
    update_labels = {}

    for r in recip_rels:
        prop_l = prop_labels[r]
        recip_prop_l = prop_labels[recip_rels[r]]
        update_labels[r] = prop_l + ' / ' + recip_prop_l

    prop_labels = {**prop_labels, **update_labels}

    # Start building the graph
    G = nx.MultiDiGraph()
    for qid, count in type_count.items():
        G.add_node(qid, count=count)

    for qid, props in type_prop_id.items():
        G.node[qid].update({k: v for k, v in props.items() if v > min_counts})
        if len(props) <= 2:
            G.node[qid].update(props)

    filt_spo, filt_spo_in = remove_reciprocal_edges(spo, spo_in, recip_rels)

    for qid, pid_conn_types in filt_spo.items():
        for pid, conn_types in pid_conn_types.items():
            # Don't add blacklisted edges
            if pid in banned_edges:
                continue
            for conn_qid, conn_count in conn_types.items():

                # must have more than Min_counts and more than filt_val * num_nodes... for either seed node...
                if (qid, pid, conn_qid) in special_edges or conn_count > min_counts:
                    G.add_edge(qid, conn_qid, key=next(keygen), count=conn_count, label=prop_labels[pid], pid=pid,
                               URL="https://www.wikidata.org/wiki/Property:" + pid)
    for qid, pid_conn_types in filt_spo_in.items():
        for pid, conn_types in pid_conn_types.items():
            for conn_qid, conn_count in conn_types.items():
                if conn_qid in filt_spo and pid in filt_spo[conn_qid] and qid in filt_spo[conn_qid][pid]:
                    # don't add duplicate edges
                    continue

                if (conn_qid, pid, qid) in special_edges or conn_count > min_counts:
                    G.add_edge(conn_qid, qid, key=next(keygen), count=conn_count, label=prop_labels[pid], pid=pid,
                               URL="https://www.wikidata.org/wiki/Property:" + pid)

    # Remove dangling edges, unless whitelisted (assume all count edges are whitelist)
    to_remove = []
    for node_id, degree in G.degree:
        if degree < 2 and node_id not in set(whitelisted_nodes).union(set(type_count.keys())):
            to_remove.append(node_id)
        elif node_id in banned_nodes:
            to_remove.append(node_id)
    for nid in to_remove:
        G.remove_node(nid)

    # Label everything
    node_labels = getConceptLabels(G.nodes())
    nx.set_node_attributes(G, {qid: "https://www.wikidata.org/wiki/" + qid for qid in node_labels}, 'URL')
    nx.set_node_attributes(G, node_labels, 'label')
    nx.set_node_attributes(G, node_labels, 'NodeLabel')
    nx.set_node_attributes(G, {qid: label + '\n' + "{:,}".format(type_count[qid]) if qid in type_count else label
                               for qid, label in node_labels.items()}, 'labelcount')

    edge_label = nx.get_edge_attributes(G, "label")
    edge_count = nx.get_edge_attributes(G, "count")
    nx.set_edge_attributes(G,
                           {k: edge_label[k] + " (" + "{:,}".format(edge_count[k]) + ")" for k in edge_label},
                           'labelcount')

    # make a multiline string for the node properties text from the external ids and counts
    exclude = {'count', 'NodeLabel', 'label', 'labelcount', 'URL', 'node_prop_text'}

    node_prop_text = dict()
    for qid in G.node:
        # Grab non-excluded props
        this_props = []
        for k, v in G.node[qid].items():
            if k not in exclude and v > G.node[qid].get('count', 0) * filt_props:
                this_props.append((prop_labels[k], v))

        # Sort via number, highest first and add
        this_props = sorted(this_props, key=lambda x: x[1], reverse=True)
        node_prop_text[qid] = '\n'.join(["{}: {:,}".format(prop, num) for prop, num in this_props])

    nx.set_node_attributes(G, node_prop_text, 'node_prop_text')

    # specify node types
    nx.set_node_attributes(G, {qid: 'detail' if node_prop_text[qid] else 'simple' for qid in node_prop_text},
                           'node_type')

    return G


def plot(G, show_box=False):
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    labels = nx.get_node_attributes(G, 'label')
    counts = nx.get_node_attributes(G, 'count')
    labels = {qid: "{}\nCount: {}".format(labels[qid], counts[qid]) if qid in counts else labels[qid] for qid in labels}
    nx.draw_networkx_labels(G, pos, labels=labels)
    nx.draw_networkx_edges(G, pos)

    edge_labels = dict([((u, v,), d.get('label', '')) for u, v, d in G.edges(data=True)])
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    if show_box:
        for qid in pos:
            x, y = pos[qid]
            label = '\n'.join({"{}: {}".format(k, v) for k, v in G.node[qid].items()})
            plt.text(x, y - 0.05, s=label, bbox=dict(facecolor='red', alpha=0.5), verticalalignment='top',
                     horizontalalignment='center')
    plt.show()


def write_graphml(G, filename):
    _write_graphml(G, filename)
    s = open(filename).read().replace('attr.type="long"', 'attr.type="int"')  # yEd nonsense
    with open(filename, "w") as f:
        f.write(s)


if __name__ == "__main__":
    min_counts = 200

    # these are the special nodes that will have their external ID counts displayed,
    # the labels aren't, outputted, only used for monitoring status
    seed_nodes = {
#        'Q12136': 'disease',
#        'Q7187': 'gene',
#        'Q8054': 'protein',
#        'Q37748': 'chromosome',
#        'Q215980': 'ribosomal RNA',
#        'Q11173': 'chemical_compound',
        'Q12140': 'medication',
        'Q28885102': 'pharmaceutical_product',
#        'Q417841': 'protein_family',
#        'Q898273': 'protein_domain',
#        'Q2996394': 'biological_process',
#        'Q14860489': 'molecular_function',
#        'Q5058355': 'cellular_component',
#        'Q3273544': 'structural_motif',
#        'Q7644128': 'supersecondary_structure',
#        'Q616005': 'binding_site',
#        'Q423026': 'active_site',
#        'Q16521': 'taxon',
##        'Q13442814': 'scientific_article',
#        'Q4936952': 'anatomical structure',
#        'Q169872': 'symptom',
##        'Q621636': 'route of admin',
        'Q15304597': 'sequence variant',
#        'Q4915012': 'biological pathway',
#        'Q50377224': 'pharmacologic action',  # Subclass
#        'Q50379781': 'therapeutic use',
#        'Q3271540': 'mechanism of action',  # Subclass
##        'Q21167512': 'chemical hazard',
##        'Q21014462': 'cell line'
    }

    # skip edge searches for: scientific article, taxon
    skip_types = {'Q13442814', 'Q16521'}

    type_count, type_prop_id, spo, spo_in = search_metagraph_from_seeds(seed_nodes, skip_types, min_counts, 200000)
    G = create_graph(type_count, type_prop_id, spo, spo_in, 200)
    filename = "tmp.graphml"
    write_graphml(G, filename)
