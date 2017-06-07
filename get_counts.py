"""
Get counts for all items we care about, and some we don't ....
"""
import functools

import requests
from networkx.readwrite.graphml import write_graphml
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

# these are the "special" nodes that will have their external ID counts displayed,
# and are the seed nodes to get everything that links off of them.
# the labels aren't used. only for me
qid_label = {
    'Q12136': 'disease',
    'Q7187': 'gene',
    'Q8054': 'protein',
    'Q37748': 'chromosome',
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
    'Q14633912': 'post-translational_protein_modification',
    'Q616005': 'binding_site',
    'Q423026': 'active_site',
    'Q16521': 'taxon',
    'Q13442814': 'scientific_article',
    'Q68685': 'metabolic pathway',
    'Q15304597': 'sequence variant'  # TODO: andra needs to change this to use instance of
}
#skip_types = {'Q13442814', 'Q16521'}


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def getConceptLabel(qid):
    return getConceptLabels((qid,))[qid]


def getConceptLabels(qids):
    out = dict()
    for chunk in chunks(list(set(qids)), 50):
        this_qids = "|".join({qid.replace("wd:", "") if qid.startswith("wd:") else qid for qid in chunk})
        params = {'action': 'wbgetentities', 'ids': this_qids, 'languages': 'en', 'format': 'json', 'props': 'labels'}
        r = requests.get("https://www.wikidata.org/w/api.php", params=params)
        print(r.url)
        r.raise_for_status()
        wd = r.json()['entities']
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
    d = {x['property']['value'].replace("http://www.wikidata.org/entity/", ""): x['propertyLabel']['value'] for x in d}
    return d


def get_type_count(qids):
    """
    For each qid, get the number of items that are instance of (types) this qid
    """
    # fails doing them all at once....
    # wds = " ".join(["wd:" + x for x in qid_label.keys()])
    dd = dict()
    for qid in tqdm(qids):
        s = """
        SELECT ?instance (COUNT(?item) as ?c) WHERE {
          values ?instance {wds}
          ?item wdt:P31 ?instance .
          ?item wdt:P31 ?instance .
        } GROUP BY ?instance
        """.replace("wds", "wd:" + qid)
        d = execute_sparql_query(s)['results']['bindings']
        d = {x['instance']['value'].replace("http://www.wikidata.org/entity/", ""): int(x['c']['value']) for x in d}
        dd.update(d)
    return dd


def get_type_edge_frequency_out(type_qid):
    """
    Another version of get_types_props. returns about the same results
    gets properties on items that are an instance of type_qid
    does not filter by prop type. You need to do that after
    """
    s = """SELECT ?property ?count WHERE {
              {
              SELECT ?property (COUNT(*) AS ?count)
                 WHERE {
                    ?item wdt:P31 wd:{xxx} .
                    ?item ?property ?value .
                   # FILTER(regex(str(?property), "http://www.wikidata.org/prop/direct"))
                   # FILTER(regex(str(?value), "http://www.wikidata.org/entity"))
                 } GROUP BY ?property
              }
            } ORDER BY DESC (?count)""".replace("{xxx}", type_qid)
    d = execute_sparql_query(s)['results']['bindings']
    r = {x['property']['value'].replace("http://www.wikidata.org/prop/direct/", ""):
             int(x['count']['value']) for x in d if 'http://www.wikidata.org/prop/direct' in x['property']['value']}
    return r


def get_type_edge_frequency_in(type_qid):
    """
    Same format as get_type_edge_frequency_out, but returns "incoming"/"reverse"/"what links here" edges
    gets properties on items that are the subjects where the object is an instance of type_qid
    does not filter by prop type. You need to do that after
    """
    s_rev = """SELECT ?property ?count WHERE {
              {
              SELECT ?property (COUNT(*) AS ?count)
                 WHERE {
                    ?value wdt:P31 wd:{xxx} .
                    ?item ?property ?value .
                   # FILTER(regex(str(?property), "http://www.wikidata.org/prop/direct"))
                 } GROUP BY ?property
              }
            } ORDER BY DESC (?count)""".replace("{xxx}", type_qid)
    d_rev = execute_sparql_query(s_rev)['results']['bindings']
    r_rev = {x['property']['value'].replace("http://www.wikidata.org/prop/direct/", ""):
                 int(x['count']['value']) for x in d_rev if
             'http://www.wikidata.org/prop/direct' in x['property']['value']}
    return r_rev


def get_types_props(type_qid, prop_type):
    """
    Given the qid of a type, get all external id props for items of this type
    ranked by usage count
    :param type_qid: QID of item type
    :param prop_type: ExternalId or WikibaseItem
    :return:
    """
    assert prop_type in {'ExternalId', 'WikibaseItem'}
    s = """
    SELECT ?property ?propertyLabel ?propertyDescription ?count WHERE {
      {
        select ?propertyclaim (COUNT(*) AS ?count) where {
          ?item wdt:P31 wd:{xxx} .
          ?item ?propertyclaim [] .
        } group by ?propertyclaim
      }
      ?property wikibase:propertyType wikibase:{yyy} .
      ?property wikibase:claim ?propertyclaim .
      SERVICE wikibase:label {bd:serviceParam wikibase:language "en" .}
    } ORDER BY DESC (?count)""".replace("{xxx}", type_qid).replace("{yyy}", prop_type)

    d = execute_sparql_query(s)['results']['bindings']
    return [(x['property']['value'].replace("http://www.wikidata.org/entity/", ""),
             x['propertyLabel']['value'],
             int(x['count']['value'])) for x in d]


def get_connecting_types_out(qid, pid, ret='label', include_subclass=False):
    """
    Given a subject item type (by its qid, ex: Q12136 (disease))
    and a property (by its pid, ex: P2176 (drug used for treatment))
    Get me the types of objects it connects to, grouped by counts
    """
    assert ret in {'label', 'qid'}
    s = """select ?type ?typeLabel (COUNT(*) AS ?count) where {
      ?item wdt:P31 wd:{xxx} .
      ?item wdt:{yyy} ?other_item .
      ?other_item wdt:P31 ?type .
      OPTIONAL { ?type rdfs:label ?typeLabel. FILTER(LANG(?typeLabel) = "en"). }
    } group by ?type ?typeLabel""".replace("{xxx}", qid).replace("{yyy}", pid)
    if include_subclass:
        s = """select ?type ?typeLabel (COUNT(*) AS ?count) where {
              ?item wdt:P31 wd:{xxx} .
              ?item wdt:{yyy} ?other_item .
              ?other_item wdt:P31|wdt:P279 ?type .  #  <------ this is the secret sauce
              OPTIONAL { ?type rdfs:label ?typeLabel. FILTER(LANG(?typeLabel) = "en"). }
            } group by ?type ?typeLabel""".replace("{xxx}", qid).replace("{yyy}", pid)
    d = execute_sparql_query(s)['results']['bindings']
    if ret == 'qid':
        return {x['type']['value'].replace("http://www.wikidata.org/entity/", ""): int(x['count']['value']) for x in d}
    else:
        return {x['typeLabel']['value']: int(x['count']['value']) for x in d}


def get_connecting_types_in(qid, pid, ret='label'):
    """
    Given a subject item type (by its qid, ex: Q12136 (disease))
    and a property (by its pid, ex: P2176 (drug used for treatment))
    Get me the types of objects that connects to it, grouped by counts
    :param qid:
    :param pid:
    :return:
    """
    assert ret in {'label', 'qid'}
    s = """select ?type ?typeLabel (COUNT(*) AS ?count) where {
      ?item wdt:P31 wd:{xxx} .
      ?other_item wdt:{yyy} ?item .
      ?other_item wdt:P31 ?type .
      OPTIONAL { ?type rdfs:label ?typeLabel. FILTER(LANG(?typeLabel) = "en"). }
    } group by ?type ?typeLabel""".replace("{xxx}", qid).replace("{yyy}", pid)
    d = execute_sparql_query(s)['results']['bindings']
    if ret == 'qid':
        return {x['type']['value'].replace("http://www.wikidata.org/entity/", ""): int(x['count']['value']) for x in d}
    else:
        return {x['typeLabel']['value']: int(x['count']['value']) for x in d}


if __name__ == "__main__":

    prop_labels = get_prop_labels()

    # get all node counts for my special types
    print("Getting type counts")
    type_count = get_type_count(qid_label)
    print("Getting type counts: Done")

    # for each types of item, get the external ID props items of this type use
    print("Getting type external ids props")
    type_prop_id = dict()
    for qid in tqdm(type_count):
        try:
            props = get_types_props(qid, 'ExternalId')
        except HTTPError:
            type_prop_id[qid] = {}
            continue
        props = {x[1]: x[2] for x in props}  # label: count
        type_prop_id[qid] = props
    print("Getting type external ids props: Done")

    # for each types of item, get the WikibaseItem props it uses
    print("Getting type item props")
    type_props_out = dict()
    for qid in tqdm(type_count):
        try:
            props = get_type_edge_frequency_out(qid)
        except HTTPError:
            type_props_out[qid] = {}
            continue
        type_props_out[qid] = props
    # and incoming
    type_props_in = dict()
    for qid in tqdm(type_count):
        try:
            props = get_type_edge_frequency_in(qid)
        except HTTPError:
            type_props_in[qid] = {}
            continue
        type_props_in[qid] = props
    print("Getting type item props: Done")

    # get the types of items that each subject -> property edge connects to
    print("Getting type item props objects")
    spo = dict()
    for qid, props in tqdm(type_props_out.items()):
        spo[qid] = dict()
        # speed this up
        props = {k: v for k, v in props.items() if v > MIN_COUNT}
        for pid, count in tqdm(props.items()):
            conn_types = get_connecting_types_out(qid, pid, ret='qid')
            spo[qid][pid] = conn_types
    # and incoming
    spo_in = dict()
    for qid, props in tqdm(type_props_in.items()):
        spo_in[qid] = dict()
        # speed this up
        props = {k: v for k, v in props.items() if v > MIN_COUNT}
        for pid, count in tqdm(props.items()):
            conn_types = get_connecting_types_in(qid, pid, ret='qid')
            spo_in[qid][pid] = conn_types
    print("Getting type item props objects: Done")

    """
    This is a little hacky, but bear with me
    Not everying in wikidata is using the "instance of" ~= type system.

    For example "chemical compounds" have 594 "cause of"s (type_props['Q11173'])
    but only to a total of 5 items that are in instance of something (spo['Q11173']['P1542'])
    Find these, and check the subclasses
    """
    print("getting special edges")
    special_edges = []
    for qid, props in type_props_out.items():
        for pid, count in props.items():
            if pid not in {'P31', 'P279', 'P680', 'P682', 'P681'} and count > MIN_COUNT and sum(spo[qid][pid].values()) < count * .75:
                #print(qid, pid, prop_labels[pid])
                special_edges.append((qid, pid))
    for qid, pid in tqdm(special_edges):
        conn_types = get_connecting_types_out(qid, pid, ret='qid', include_subclass=True)
        if conn_types:
            print(qid, pid, prop_labels[pid])
            conn_types = {k: v for k, v in conn_types.items() if v > MIN_COUNT}
            print(conn_types)
            spo[qid][pid] = conn_types
    print("getting special edges: Done")
    ############
    # construct the network
    ############
    G = nx.MultiDiGraph()
    for qid, count in type_count.items():
        label = qid_label[qid]
        G.add_node(qid, {'count': count, 'label': label})
    for qid, props in type_prop_id.items():
        G.node[qid].update({k: v for k, v in props.items() if v > MIN_COUNT})
        if len(props) <= 2:
            G.node[qid].update(props)
    for qid, pid_conn_types in spo.items():
        for pid, conn_types in pid_conn_types.items():
            for conn_qid, conn_count in conn_types.items():
                if conn_count > MIN_COUNT:
                    G.add_edge(qid, conn_qid, count=conn_count, label=prop_labels[pid])
    for qid, pid_conn_types in spo_in.items():
        for pid, conn_types in pid_conn_types.items():
            for conn_qid, conn_count in conn_types.items():
                if conn_count > MIN_COUNT:
                    G.add_edge(conn_qid, qid, count=conn_count, label=prop_labels[pid])
    node_labels = getConceptLabels(G.nodes())
    nx.set_node_attributes(G, 'label', node_labels)
    nx.set_node_attributes(G, 'NodeLabel', node_labels)
    nx.set_node_attributes(G, 'labelcount',
                           {qid: label + '\n' + str(type_count[qid]) if qid in type_count else label for qid, label in
                            node_labels.items()})

    edge_label = nx.get_edge_attributes(G, "label")
    edge_count = nx.get_edge_attributes(G, "count")
    nx.set_edge_attributes(G, 'labelcount', {k: edge_label[k] + " (" + str(edge_count[k]) + ")" for k in edge_label})

    # make a multiline string for the node properties text from the external ids and counts
    exclude = {'count', 'NodeLabel', 'label', 'labelcount'}
    node_prop_text = {qid: '\n'.join({"{}: {}".format(k, v) for k, v in G.node[qid].items() if k not in exclude}) for qid in
                      G.node}
    nx.set_node_attributes(G, 'node_prop_text', node_prop_text)

    # specify node types
    nx.set_node_attributes(G, 'node_type', {qid: 'detail' if node_prop_text[qid] else 'simple' for qid in node_prop_text})

    write_graphml(G, "tmp.graphml")
    s = open("tmp.graphml").read().replace('attr.type="long"', 'attr.type="int"')  # yEd nonsense
    with open("tmp.graphml", "w") as f:
        f.write(s)


def plot(G, show_box = False):
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