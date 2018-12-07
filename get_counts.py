"""
Get counts for all items we care about, and some we don't ....
"""
import functools

import requests
from networkx.readwrite.graphml import write_graphml as _write_graphml
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
execute_sparql_query = functools.partial(execute_sparql_query,
                                         endpoint="http://avalanche.scripps.edu:9999/bigdata/sparql")

MIN_COUNT = 200

# these are the special nodes that will have their external ID counts displayed,
# the labels aren't used. only for my sanity
attribute_nodes = {
    'Q12136': 'disease',
    'Q7187': 'gene',
    'Q8054': 'protein',
    'Q37748': 'chromosome',
    'Q215980': 'ribosomal RNA',
    'Q11173': 'chemical_compound',
    'Q12140': 'pharmaceutical_drug',
    'Q28885102': 'pharmaceutical_product',
    'Q417841': 'protein_family',
    'Q898273': 'protein_domain',
    'Q2996394': 'biological_process',
    'Q14860489': 'molecular_function',
    'Q5058355': 'cellular_component',
    'Q3273544': 'structural_motif',
    'Q7644128': 'supersecondary_structure',
    'Q616005': 'binding_site',
    'Q423026': 'active_site',
    'Q16521': 'taxon',
    'Q13442814': 'scientific_article',
    'Q4936952': 'anatomical structure',
    'Q169872': 'symptom',
    'Q621636': 'route of admin',
    'Q4936952': 'anatomical structure',
    'Q169872': 'symptom',
    'Q15304597': 'sequence variant',
    'Q4915012': 'biological pathway',
}

# subclass nodes
# these are "special" because they don't use the instance of = type system, because they were done by others
# we'll have to get items that are subclass of these
subclass_nodes = {
                  # niosh symptoms are mess, half are not instance of anything, a quarter have no instance of or subclass # http://tinyurl.com/y98bk69x
                  'Q21167512': 'chemical hazard',  # this too. look into subclass* hazard (Q1132455)
                  }

# seed nodes get everything that links off of them.
skip_types = {'Q13442814', 'Q16521'}
# skip: scientific article, taxon
seed_nodes = dict(attribute_nodes.items() | subclass_nodes.items())
seed_nodes = {k: v for k, v in seed_nodes.items() if k not in skip_types}

# instance of subject, subclass of object
special_edges = [('Q11173', 'P1542', 'Q21167512'),  # chemical, cause of, chemical hazard
                 ('Q12136', 'P780', 'Q169872'),  # disease, symptom, symptom
                 ('Q12136', 'P780', 'Q1441305'),  # disease, symptom, medical sign
                 ('Q21167512', 'P780', 'Q169872')]  # chemical hazard, symptom, symptom


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
        out.update({k: v['labels']['en']['value'] for k, v in wd.items()})
    return out


def get_prop_labels():
    """ returns a dict of labels for all properties in wikidata """
    s = """
    SELECT DISTINCT ?property ?propertyLabel
    WHERE {
        ?property a wikibase:Property .
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }"""
    d = execute_sparql_query(s)['results']['bindings']
    d = {x['property']['value'].replace("http://www.wikidata.org/entity/", ""):
                x['propertyLabel']['value'] for x in d}
    return d


def get_type_count(qids, use_subclass=False):
    """
    For each qid, get the number of items that are instance of (types) this qid
    """
    p = "P279" if use_subclass else "P31"
    dd = dict()
    for qid in tqdm(qids):
        s = """
        SELECT (COUNT(?item) as ?c) WHERE {
          ?item wdt:{p} {wds}
        }
        """.replace("{wds}", "wd:" + qid).replace("{p}", p)
        d = execute_sparql_query(s)['results']['bindings']
        d = {qid: int(x['c']['value']) for x in d}
        dd.update(d)
    return dd


def get_type_edge_frequency(type_qid, direction='out', use_subclass=False):
    """
    gets properties on items that are an instance of type_qid. returns all kinds of props, (i.e. including
        externalid and wditem). Filtering by item type causes query timeouts
    if direction = "in": returns "incoming"/"reverse"/"what links here" edges, gets properties on items that are the
        subjects where the object is an instance of type_qid
    """
    assert direction in {'in', 'out'}
    p = "P279" if use_subclass else "P31"
    subject = "subject"
    if direction == "in":
        subject = "object"
    s = """SELECT ?property ?count WHERE {
              {
              SELECT ?property (COUNT(*) AS ?count) WHERE {
                    ?{subject} wdt:{p} wd:{xxx} .
                    ?subject ?property ?object .
                 } GROUP BY ?property
              }
            } ORDER BY DESC (?count)""".replace("{xxx}", type_qid).replace("{p}", p).replace("{subject}", subject)
    d = execute_sparql_query(s)['results']['bindings']
    r = {x['property']['value'].split('/')[-1]:
             int(x['count']['value']) for x in d if 'http://www.wikidata.org/prop/direct' in x['property']['value']}
    return r


def get_external_ids(type_qid, use_subclass=False):
    """
    Given the qid of a type, get all external id props for items of this type
    ranked by usage count
    :param type_qid: QID of item type
    """
    p = "P279" if use_subclass else "P31"
    s = """
    SELECT ?property ?propertyLabel ?propertyDescription ?count WHERE {
      {
        select ?propertyclaim (COUNT(*) AS ?count) where {
          ?item wdt:{p} wd:{type_qid} .
          ?item ?propertyclaim [] .
        } group by ?propertyclaim
      }
      ?property wikibase:propertyType wikibase:ExternalId .
      ?property wikibase:claim ?propertyclaim .
      SERVICE wikibase:label {bd:serviceParam wikibase:language "en" .}
    } ORDER BY DESC (?count)""".replace("{type_qid}", type_qid).replace("{p}", p)

    d = execute_sparql_query(s)['results']['bindings']
    return [(x['property']['value'].replace("http://www.wikidata.org/entity/", ""),
             x['propertyLabel']['value'],
             int(x['count']['value'])) for x in d]


def get_connecting_types(qid, pid, direction='out', use_subclass_subject=False, include_subclass_object=False):
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
    assert direction in {'in', 'out'}
    p_sub = "P279" if use_subclass_subject else "P31"
    left, right = 'subject', 'object'
    if direction == 'in':
        left, right = right, left
    p_obj = "wdt:P31|wdt:P279" if include_subclass_object else "wdt:P31"

    s = """select ?type (COUNT(*) AS ?count) where {{
      ?subject wdt:{p_sub} wd:{qid} .
      ?{left} wdt:{pid} ?{right} .
      ?object {p_obj} ?type .
    }} group by ?type""".format(p_sub=p_sub, qid=qid, pid=pid, p_obj=p_obj, left=left, right=right)
    d = execute_sparql_query(s)['results']['bindings']
    return {x['type']['value'].replace("http://www.wikidata.org/entity/", ""): int(x['count']['value']) for x in d}


def generate_graph_data():
    # get all node counts for my special types
    print("Getting type counts")
    type_count = get_type_count(attribute_nodes)
    type_count_sub = get_type_count(subclass_nodes, use_subclass=True)
    type_count.update(type_count_sub)
    print("Getting type counts: Done")

    # for each types of item, get the external ID props items of this type use
    print("Getting type external ids props")
    type_prop_id = dict()
    for qid in tqdm(attribute_nodes):
        props = get_external_ids(qid)
        props = {x[0]: x[2] for x in props}  # id: count
        type_prop_id[qid] = props
    for qid in tqdm(subclass_nodes):
        props = get_external_ids(qid, use_subclass=True)
        props = {x[0]: x[2] for x in props}  # id: count
        type_prop_id[qid] = props
    print("Getting type external ids props: Done")

    # for each types of item, get the WikibaseItem props it uses (using the seed nodes)
    print("Getting outgoing props for each type")
    type_props_out = dict()
    for qid in tqdm(seed_nodes):
        props = get_type_edge_frequency(qid, direction='out')
        # remove external id props
        props = {k: v for k, v in props.items() if k not in type_prop_id[qid]}
        type_props_out[qid] = props
    # and incoming
    print("Getting incoming props for each type")
    type_props_in = dict()
    for qid in tqdm(seed_nodes):
        props = get_type_edge_frequency(qid, direction='in')
        type_props_in[qid] = props
    print("Done")

    # same as above, special dealing with subclass
    for qid in subclass_nodes:
        props = get_type_edge_frequency(qid, direction='out', use_subclass=True)
        # remove external id props
        props = {k: v for k, v in props.items() if k not in type_prop_id[qid]}
        type_props_out[qid] = props
    for qid in subclass_nodes:
        props = get_type_edge_frequency(qid, direction='in', use_subclass=True)
        type_props_in[qid] = props

    # get the types of items that each subject -> property edge connects to
    print("Getting type of objects each (subject, predicate) connects to")
    spo = dict()
    t = tqdm(type_props_out.items())
    for qid, props in t:
        t.set_description(seed_nodes[qid])
        t.refresh()
        spo[qid] = dict()
        props = {k: v for k, v in props.items() if v > MIN_COUNT}
        for pid, count in tqdm(props.items()):
            conn_types = get_connecting_types(qid, pid, direction='out')
            spo[qid][pid] = conn_types
    # and incoming
    print("Getting type of objects each (subject, predicate) connects from")
    spo_in = dict()
    t = tqdm(type_props_in.items())
    for qid, props in t:
        t.set_description(seed_nodes[qid])
        t.refresh()
        spo_in[qid] = dict()
        props = {k: v for k, v in props.items() if v > MIN_COUNT}
        for pid, count in tqdm(props.items()):
            conn_types = get_connecting_types(qid, pid, direction='in')
            spo_in[qid][pid] = conn_types
    print("Done")

    ## special dealing with subclass
    print("Getting type of objects each (subject, predicate) connects to SUBCLASS")
    t = tqdm(type_props_out.items())
    for qid, props in t:
        if qid not in subclass_nodes:
            continue
        t.set_description(seed_nodes[qid])
        t.refresh()
        spo[qid] = dict()
        props = {k: v for k, v in props.items() if v > MIN_COUNT}
        for pid, count in tqdm(props.items()):
            conn_types = get_connecting_types(qid, pid, direction='out', use_subclass_subject=True)
            spo[qid][pid] = conn_types
    # and incoming
    print("Getting type of objects each (subject, predicate) connects from SUBCLASS")
    t = tqdm(type_props_in.items())
    for qid, props in t:
        if qid not in subclass_nodes:
            continue
        t.set_description(seed_nodes[qid])
        t.refresh()
        spo_in[qid] = dict()
        props = {k: v for k, v in props.items() if v > MIN_COUNT}
        for pid, count in tqdm(props.items()):
            conn_types = get_connecting_types(qid, pid, direction='in', use_subclass_subject=True)
            spo_in[qid][pid] = conn_types

    print("really special edges")
    for qid, pid, qid2 in special_edges:
        conn_types = get_connecting_types(qid, pid, direction='out', use_subclass_subject=True,
                                          include_subclass_object=True)
        print(conn_types[qid2])
        spo[qid][pid] = conn_types
    print("Done")
    return type_count, type_prop_id, spo, spo_in


def create_graph(type_count, type_prop_id, spo, spo_in):
    ############
    # construct the network
    ############
    prop_labels = get_prop_labels()

    G = nx.MultiDiGraph()
    for qid, count in type_count.items():
        G.add_node(qid, {'count': count})
    for qid, props in type_prop_id.items():
        G.node[qid].update({k: v for k, v in props.items() if v > MIN_COUNT})
        if len(props) <= 2:
            G.node[qid].update(props)
    for qid, pid_conn_types in spo.items():
        for pid, conn_types in pid_conn_types.items():
            for conn_qid, conn_count in conn_types.items():
                if (qid, pid, conn_qid) in special_edges or conn_count > MIN_COUNT:
                    G.add_edge(qid, conn_qid, count=conn_count, label=prop_labels[pid], pid=pid,
                               URL="https://www.wikidata.org/wiki/Property:" + pid)
    for qid, pid_conn_types in spo_in.items():
        for pid, conn_types in pid_conn_types.items():
            for conn_qid, conn_count in conn_types.items():
                if conn_qid in spo and pid in spo[conn_qid] and qid in spo[conn_qid][pid]:
                    # don't add duplicate edges
                    continue
                if conn_count > MIN_COUNT:
                    G.add_edge(conn_qid, qid, count=conn_count, label=prop_labels[pid], pid=pid,
                               URL="https://www.wikidata.org/wiki/Property:" + pid)
    node_labels = getConceptLabels(G.nodes())
    nx.set_node_attributes(G, 'URL', {qid: "https://www.wikidata.org/wiki/" + qid for qid in node_labels})
    nx.set_node_attributes(G, 'label', node_labels)
    nx.set_node_attributes(G, 'NodeLabel', node_labels)
    nx.set_node_attributes(G, 'labelcount',
                           {qid: label + '\n' + str(type_count[qid]) if qid in type_count else label for qid, label in
                            node_labels.items()})

    edge_label = nx.get_edge_attributes(G, "label")
    edge_count = nx.get_edge_attributes(G, "count")
    nx.set_edge_attributes(G, 'labelcount', {k: edge_label[k] + " (" + str(edge_count[k]) + ")" for k in edge_label})

    # make a multiline string for the node properties text from the external ids and counts
    exclude = {'count', 'NodeLabel', 'label', 'labelcount', 'URL', 'node_prop_text'}
    node_prop_text = {
    qid: '\n'.join({"{}: {}".format(prop_labels[k], v) for k, v in G.node[qid].items() if k not in exclude}) for
    qid in G.node}
    nx.set_node_attributes(G, 'node_prop_text', node_prop_text)

    # specify node types
    nx.set_node_attributes(G, 'node_type',
                           {qid: 'detail' if node_prop_text[qid] else 'simple' for qid in node_prop_text})

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
    type_count, type_prop_id, spo, spo_in = generate_graph_data()
    G = create_graph(type_count, type_prop_id, spo, spo_in)
    filename = "tmp.graphml"
    write_graphml(G, filename)
