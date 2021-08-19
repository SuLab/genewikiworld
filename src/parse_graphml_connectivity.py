import argparse
import pandas as pd
import get_counts as gc
import update_graphml as ug

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

def get_info_from_graphml(filename, outname=None, endpoint=None):

    query_list = dict()

    # Use the desired endpoint
    if endpoint is not None:
        gc.change_endpoint(endpoint)

    # Read the graphml file and break into nodes and edged
    tree = ug.read_graphml(filename)
    root = tree.getroot()
    graph = ug.get_graph(root)
    nodes = ug.get_nodes(graph)
    edges = ug.get_edges(graph)

    # Get info specific to this WikiData items from the graph
    n_id_map, e_id_map = ug.get_node_edge_attrib_mappers(root)
    n_to_qid = ug.get_node_id_to_qid(nodes, n_id_map)
    node_info = ug.get_node_info_to_update(nodes, n_id_map)
    edge_info = ug.get_edge_info_to_update(edges, n_to_qid, e_id_map)

    prop_names = gc.get_prop_labels()
    node_names = {k: v.get('NodeLabel') for k, v in node_info.items()}


    # Initialize the variables to collect data
    n_ids = []
    n_names = []
    p_ids = []
    p_names = []
    counts = []
    # Loop through and extract all node property information
    for n_id, n_info in node_info.items():
        for prop, count in n_info.get('props', dict()).items():
            n_ids.append(n_id)
            n_names.append(node_names.get(n_id))
            p_ids.append(prop)
            p_names.append(prop_names.get(prop))
            counts.append(count)
    # Compile results to DataFrame
    nodes_out = pd.DataFrame({'subject_type_name': n_names, 'subject_type_qid': n_ids,
                              'property_name': p_names, 'property_pid': p_ids,
                              'count': counts})
    # sort the nodes by count for easier comparison
    nodes_out = nodes_out.sort_values(['subject_type_name', 'count'], ascending=[True, False])

    # Initialize edge information collection
    n1_ids = []
    n1_names = []
    n2_ids = []
    n2_names = []
    p_ids = []
    p_names = []
    for edge_key in edge_info.keys():
        n1_id = edge_key[0]
        p_id = edge_key[1]
        n2_id = edge_key[2]

        n1_ids.append(n1_id)
        n1_names.append(node_names.get(n1_id))
        p_ids.append(p_id)
        p_names.append(prop_names.get(p_id))
        n2_ids.append(n2_id)
        n2_names.append(node_names.get(n2_id))

        # Get revierse edge info if a reciprical relationship
        if p_id in recip_rels:
            n1_ids.append(n2_id)
            n1_names.append(node_names.get(n2_id))
            p_ids.append(recip_rels[p_id])
            p_names.append(prop_names.get(recip_rels[p_id]))
            n2_ids.append(n1_id)
            n2_names.append(node_names.get(n1_id))

    edges_out = pd.DataFrame({'subject_type_name': n1_names, 'subject_type_qid': n1_ids,
                              'property_name': p_names, 'property_pid': p_ids,
                              'object_type_name': n2_names, 'object_type_qid': n2_ids})

    out = pd.concat([nodes_out, edges_out], sort=False).drop('count', axis=1)
    if outname is None:
        outname = 'query_info.csv'
    out.to_csv(outname, index=False)


if __name__ == "__main__":
    # Command line Parsing
    parser = argparse.ArgumentParser(description='Parse a complete .graphml to get arguments for querying data'+
                                                 ' provenance. (input to get_prov_counts.py)')
    parser.add_argument('filename', help="The .graphml to parse for subject, predicate, object and property" + \
                                          " identifiers and information", type=str)
    parser.add_argument('-o', '--outname', help="The name of the output file. (Default 'query_info.csv')",
                        type=str, default=None)
    parser.add_argument('-e', '--endpoint', help='Use a wikibase endpoint other than standard wikidata', type=str,
                        default=None)

    #Unpack the CLI args
    args = parser.parse_args()
    filename = args.filename
    outname = args.outname
    endpoint = args.endpoint

    # run the pipeline
    get_info_from_graphml(filename, outname=outname, endpoint=None)

