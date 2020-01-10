import datetime
import pandas as pd
import get_counts as gc
from collections import defaultdict


LOGSTR = ""

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

    # If an edge query, only query for certain biological edges 1 time
    if 'biological_edge' in query_df:
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


def get_reference_info(query_info_file, outname, logfile=None):

    query_info = pd.read_csv(query_info_file)

    prop_qs = query_info[query_info['object_type_qid'].isnull()]
    edge_qs = query_info[~query_info['object_type_qid'].isnull()]

    prop_res = run_prov_queries(prop_qs)
    prop_res.to_csv(outname, index=False)
    prop_res = pd.DataFrame()

    edge_res = run_prov_queries(edge_qs)
    pd.concat([prop_res, edge_res], sort=False, ignore_index=True).to_csv(outname, index=False)

    write_log(logfile)


if __name__ == '__main__':
    get_reference_info('query_info.csv', 'test_prov.csv')

