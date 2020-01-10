import argparse
import datetime
import pandas as pd
import get_counts as gc
from collections import defaultdict


LOGSTR = ""

biological_properties = ['P780', # symptoms
                         'P2176', # drug used for treatment
                         'P2175', # medical condition treated
                         'P2293', # genetic association
                         'P927', # anatomical location
                         'P703', # found in taxon
                         'P684', # ortholog
                         'P1057', # chromosome
                         'P680', # molecular function
                         'P681', # cell component
                         'P702', # encoded by
                         'P688', # encodes
                         'P769', # significant drug interaction
                         'P3781', # has active ingredient
                         'P3780', # active ingredient in
                         'P4044', # therapeutic area
                         'P128', # regulates (molecular biology)
                         'P3433', # biological variant of
                         'P3354', # positive therapeutic predictor
                         'P3355', # negative therapeutic predictor
                         ]


def write_log(logfile=None):
    global LOGSTR

    if logfile is None:
        now = datetime.datetime.now()
        now = now.strftime('%Y-%M-%d_%H-%m-%S')
        logfile = 'failed_query_log_' + now + '.txt'

    if LOGSTR:
        print('Failed Queries reported, saving log file')
        with open(logfile, 'w') as f_out:
            f_out.write(LOGSTR)


def get_property_prov_counts(qid, pid, use_subclass=False, extend_subclass=False):

    global LOGSTR

    p_subj = gc.determine_p(use_subclass, extend_subclass)

    prop_query = """
    SELECT ?ref ?refLabel ?count WHERE {
      {SELECT ?ref  (COUNT(*) AS ?count) WHERE {
        SELECT DISTINCT ?item ?xref ?ref WHERE  {
          ?item {p_subj} wd:{qid}.
          ?item p:{pid} [ps:{pid} ?xref;
                        prov:wasDerivedFrom
                        [pr:P248 ?ref;]
                       ]
          }} GROUP BY ?ref
       }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }

    }ORDER BY DESC (?count)""".replace('{p_subj}', p_subj).replace('{qid}', qid).replace('{pid}', pid)
    #print(prop_query)

    try:
        d = gc.execute_sparql_query(prop_query)['results']['bindings']
    except:
        print("***** FAILED SPARQL *****")
        print("Item QID: {}\tProp PID: {}\n".format(qid, pid))
        d = []
        LOGSTR += 'Node Property Reference Query:'
        LOGSTR += prop_query + '\n\n'

    return [(x['ref']['value'].replace("http://www.wikidata.org/entity/", ""),
             x['refLabel']['value'],
             int(x['count']['value'])) for x in d]


def get_edge_prov_counts(qid, pid, o_qid, use_subclass_sub=False, extend_subclass_sub=False,
                         use_subclass_obj=False, extend_subclass_obj=False, biological_edge=False):

    global LOGSTR

    p_subj = gc.determine_p(use_subclass_sub, extend_subclass_sub)
    p_obj = gc.determine_p(use_subclass_obj, extend_subclass_obj)

    if biological_edge:
        bio_edge = ""
    else:
        bio_edge = ("""?item {p_subj} wd:{qid}.
          ?obj {p_obj} wd:{o_qid}.""".replace('{p_subj}', p_subj).replace('{qid}', qid)
                                    .replace('{p_obj}', p_obj).replace('{o_qid}', o_qid))


    edge_query = """
    SELECT ?ref ?refLabel ?count WHERE {
      {SELECT ?ref  (COUNT(*) AS ?count) WHERE {
        SELECT DISTINCT ?item ?obj ?ref WHERE  {
          {bio_edge}
          ?item p:{pid} [ps:{pid} ?obj;
                        prov:wasDerivedFrom
                        [pr:P248 ?ref;]
                       ]
          }} GROUP BY ?ref
       }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }

        }ORDER BY DESC (?count)""".replace('{pid}', pid).replace('{bio_edge}', bio_edge)

    #print(edge_query)
    try:
        d = gc.execute_sparql_query(edge_query)['results']['bindings']
    except:
        print("***** FAILED SPARQL *****")
        print("Item QID: {}\tProp PID: {} Obj QID: {}\n".format(qid, pid, o_qid))
        d = []
        LOGSTR += 'Edge Reference Query:'
        LOGSTR += edge_query + '\n\n'

    return [(x['ref']['value'].replace("http://www.wikidata.org/entity/", ""),
             x['refLabel']['value'],
             int(x['count']['value'])) for x in d]


def run_prov_queries(input_df):

    # Drop empty columns
    query_df = input_df.T.dropna(how='all').T
    c_order = query_df.columns.tolist()

    # Determine the query type
    edge_query = 'object_type_qid' in query_df

    # For edge queries, biological edges will only be between to biomdeical typed nodes
    # So we can get provinence for all instances of that property
    if edge_query and 'biological_edge' not in query_df:
        query_df['biological_edge'] = query_df['property_pid'].apply(lambda p: p in biological_properties)

    # Make sure we're only querying these biological properties one time
    if edge_query:
        bio_edges = query_df.query('biological_edge')
        not_bio_edges = query_df.query('biological_edge != True')
        bio_edges = bio_edges.drop_duplicates(subset=['property_pid']).copy()

        # We'll query these on property only, so remove the subject and object type specifications
        for c in bio_edges.columns:
            if '_type_' in c:
                bio_edges[c] = float('nan')
        query_df = pd.concat([not_bio_edges, bio_edges], sort=False, ignore_index=True)


    # Initialize the columns in the final dataframe
    out_data = defaultdict(list)

    for row in query_df.itertuples(index=False):
        qid = row.subject_type_qid
        pid = row.property_pid

        # Run the query
        if not edge_query:
            sub = gc.is_subclass(qid)
            prov_counts = get_property_prov_counts(qid, pid, sub)
        else:
            o_qid = row.object_type_qid
            bio_edge = row.biological_edge if 'biological_edge' in query_df else False
            # Bio edges have 'nan' qids, so is_subclass() will fail, so skip in these cases
            if not bio_edge:
                sub = gc.is_subclass(qid)
                o_sub = gc.is_subclass(o_qid)
            else:
                sub = False
                o_sub = False
            prov_counts = get_edge_prov_counts(qid, pid, o_qid, use_subclass_sub=sub,
                                               use_subclass_obj=o_sub, biological_edge=bio_edge)
        # Print for debug purposes
        print(prov_counts)
        # Compile the results
        for ref, ref_name, count in prov_counts:
            # Sometimes get useless Pids
            if ref == ref_name:
                continue

            # Add new row to output
            for row_name, row_val in row._asdict().items():
                out_data[row_name].append(row_val)
            out_data['reference_name'].append(ref_name)
            out_data['reference_id'].append(ref)
            out_data['count'].append(count)

    # Make sure at least some results were found
    if out_data:
        # format output lists to DataFrame
        prov_out = pd.DataFrame(out_data)[c_order + ['reference_name', 'reference_id', 'count']]
    else:
        prov_out = pd.DataFrame(out_data)
    return prov_out


def filter_results(results_df, absolute_min=10, filt_level=0.05):
    """
    For a given minimum for group and given filter level, a subject predicat object group (or just subject predicate
    if there is no object) will be removed it has fewer than the absolute_min or fewer than filt_level* the groups max.

    if count >= absolute_min and count > max_for_group * filt_level: keep
    """

    # Quick check to ensure filtering is required
    if absolute_min == 0 and filt_level == 0:
        return results_df

    # if only filtering by counts, a simple query is faster
    if filt_level == 0:
        return results_df.query('count >= @absolute_min').reset_index(drop=True)

    # Determine if an edge query or a prop query by column names
    group_on = ['subject_type_qid', 'property_pid']
    if 'object_type_qid' in results_df:
        group_on.append('object_type_qid')

    # Group the resuts to determine the filtering cutoffs
    grouped = results_df.groupby(group_on)
    max_for_group = grouped['count'].apply(max)
    group_cutoffs = (max_for_group * filt_level).to_dict()

    # Perform the filtering
    filtered = []
    for row in results_df.itertuples(index=False):
        if row.count > group_cutoffs[tuple(getattr(row, g) for g in group_on)] and row.count >= absolute_min:
            filtered.append(row)

    return pd.DataFrame(filtered)


def get_reference_info(query_info_file, outname=None, endpoint=None, logfile=None, absolute_min=10, filt_level=0.05):
    """
    Run the pipline to get counts for the various references.

    Set absolute min and filt level to 0 if no filtering desired.
    """
    if outname is None:
        outname = "prov_counts.csv"

    if endpoint is not None:
        gc.change_endpound(endpoint)

    query_info = pd.read_csv(query_info_file)
    out_cols = query_info.columns.tolist() + ['reference_name', 'reference_id', 'count']

    # Determine the edge vs property queries
    prop_qs = query_info[query_info['object_type_qid'].isnull()]
    edge_qs = query_info[~query_info['object_type_qid'].isnull()]

    # Run the proerpty queries
    prop_res = run_prov_queries(prop_qs)
    prop_res = filter_results(prop_res, absolute_min, filt_level)

    # Run the edge queries
    edge_res = run_prov_queries(edge_qs).fillna('None')
    edge_res = filter_results(edge_res, absolute_min, filt_level)

    # Write out the results
    pd.concat([prop_res, edge_res], sort=False, ignore_index=True)[out_cols].to_csv(outname, index=False)
    write_log(logfile)


if __name__ == '__main__':

    # Command line Parsing
    parser = argparse.ArgumentParser(description='Query wikidata for data provinence')
    parser.add_argument('filename', help="The .csv with query information (output of " + \
                                          "parse_graphml_connectivity.py)", type=str)
    parser.add_argument('-o', '--outname', help="The name of the output file. (Default 'prov_counts.csv')",
                        type=str, default=None)
    parser.add_argument('-l', '--logfile', help="Filename for log of failed queries. " + \
                                                "Unique filenmae will be used if none passed", type=str, default=None)
    parser.add_argument('-m', '--absolute_min', help="The mininum nubmer of counts a reference must have for a given"+
                            " group, to be included (default 10)", type=int, default=10)
    parser.add_argument('-f', '--filt_level', help="The fraction of the max counts for a group that a reference must"+
                            " must have to be included (default 0.05)", type=float, default=0.05)
    parser.add_argument('-e', '--endpoint', help='Use a wikibase endpoint other than standard wikidata', type=str,
                        default=None)

    # Unpack commandline arguments
    args = parser.parse_args()
    filename = args.filename
    outname = args.outname
    logfile = args.logfile
    absolute_min = args.absolute_min
    filt_level = args.filt_level
    endpoint = args.endpoint

    get_reference_info(filename, outname, endpoint, logfile, absolute_min, filt_level)

