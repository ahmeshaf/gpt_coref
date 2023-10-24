from collections import defaultdict
from coval.eval.evaluator import evaluate_documents as evaluate
from coval.eval.evaluator import muc, b_cubed, ceafe, lea
from coval.conll.reader import get_coref_infos
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm
from typing import Optional

import copy
import itertools
import numpy as np
import os
import timeit
import typer

from .utils import load_tasks


app = typer.Typer()

# ----------------------- Helpers ------------------------- #


def generate_key_file(coref_map_tuples, name, out_dir, out_file_path):
    """

    Parameters
    ----------
    coref_map_tuples: list
    name: str
    out_dir: str
    out_file_path: str

    Returns
    -------
    None
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    clus_to_int = {}
    clus_number = 0
    with open(out_file_path, "w") as of:
        of.write("#begin document (%s);\n" % name)
        for i, map_ in enumerate(coref_map_tuples):
            en_id = map_[0]
            clus_id = map_[1]
            if clus_id in clus_to_int:
                clus_int = clus_to_int[clus_id]
            else:
                clus_to_int[clus_id] = clus_number
                clus_number += 1
                clus_int = clus_to_int[clus_id]
            of.write("%s\t0\t%d\t%s\t(%d)\n" % (name, i, en_id, clus_int))
        of.write("#end document\n")


def run_coreference_results(gold_clusters, predicted_clusters, method_name):
    gold_key_file = f"evt_gold.keyfile"
    generate_key_file(gold_clusters, "evt", "./", gold_key_file)
    system_key_file = "./evt_annotated.keyfile"
    generate_key_file(predicted_clusters, "evt", "./", system_key_file)

    def read(key, response):
        return get_coref_infos("%s" % key, "%s" % response, False, False, True)

    doc = read(gold_key_file, system_key_file)
    mr, mp, mf = np.round(np.round(evaluate(doc, muc), 3) * 100, 1)
    br, bp, bf = np.round(np.round(evaluate(doc, b_cubed), 3) * 100, 1)
    cp, cr, cf = np.round(np.round(evaluate(doc, ceafe), 3) * 100, 1)
    lr, lp, lf = np.round(np.round(evaluate(doc, lea), 3) * 100, 1)

    recll = np.round((mr + br) / 2, 1)
    precision = np.round((mp + bp) / 2, 1)
    connl = np.round((mf + bf + cf) / 3, 1)

    print(f"{method_name} &&", recll, "&", precision, "&", connl, "\\\\")


def c_clustering(sparse_matrix, m_ids):
    _, labels = connected_components(sparse_matrix, return_labels=True)
    return [(m, l) for m, l in zip(m_ids, labels)]


def get_sparse_matrix(m2i, tasks):
    eid2m_id = defaultdict(set)
    for m_id, e_ids in tasks:
        for e_id in e_ids:
            eid2m_id[e_id].add(m_id)

    row = []
    col = []
    data = []
    for m_ids in eid2m_id.values():
        for m1 in m_ids:
            for m2 in m_ids:
                if m1 in m2i and m2 in m2i:
                    row.append(m2i[m1])
                    col.append(m2i[m2])
                    data.append(True)
    n = len(m2i)
    sparse_matrix = csr_matrix(
        (data, (np.array(row), np.array(col))), shape=(n, n), dtype=np.int8
    )
    return sparse_matrix


def and_clustering(tasks_a, tasks_b):
    """
    Clusters are of the form (mention_id, EIDs)
    :param tasks_a:
    :param tasks_b:
    :return:
    """
    m_ids_ordered = [m for m, _ in tasks_a]
    m2i = {m: i for i, m in enumerate(m_ids_ordered)}

    sparse_a = get_sparse_matrix(m2i, tasks_a)
    sparse_b = get_sparse_matrix(m2i, tasks_b)

    sparse_a_and_b = sparse_a.multiply(sparse_b)
    return c_clustering(sparse_a_and_b, m_ids_ordered)


def or_clustering(tasks_a, tasks_b):
    m_ids_ordered = [m for m, _ in tasks_a]
    m2i = {m: i for i, m in enumerate(m_ids_ordered)}

    sparse_a = get_sparse_matrix(m2i, tasks_a)
    sparse_b = get_sparse_matrix(m2i, tasks_b)

    sparse_a_or_b = sparse_a + sparse_b
    return c_clustering(sparse_a_or_b, m_ids_ordered)


def a_clustering(tasks_a):
    m_ids_ordered = [m for m, _ in tasks_a]
    m2i = {m: i for i, m in enumerate(m_ids_ordered)}

    sparse_a = get_sparse_matrix(m2i, tasks_a)
    return c_clustering(sparse_a, m_ids_ordered)


def add_eid(task, syn_map, sent_rs_id2task):
    doc_id = task["doc_id"]
    sentence_id = task["sentence_id"]
    if "EIDs" not in task:
        task["EIDs"] = set()
        arg0_rs_arg1s = itertools.product(
            task["arg0"], [task["roleset_id"]], task["arg1"]
        )
        for arg0, rs, arg1 in arg0_rs_arg1s:
            if len(syn_map) and rs in syn_map:
                rs = syn_map[rs]
            if (doc_id, sentence_id, arg1) in sent_rs_id2task:
                add_eid(
                    sent_rs_id2task[doc_id, sentence_id, arg1][0],
                    syn_map,
                    sent_rs_id2task,
                )
            eid_curr = (arg0, rs, arg1)

            if (doc_id, sentence_id, arg1) not in sent_rs_id2task:
                task["EIDs"].add(eid_curr)
            else:
                hop_eids = set(sent_rs_id2task[(doc_id, sentence_id, arg1)][0]["EIDs"])
                for eid in hop_eids:
                    for i in range(0, len(eid) - 2, 2):
                        task["EIDs"].add((arg0, rs) + eid[i:])


# def generate_eids(tasks, syn_map):


def generate_eids(tasks, syn_map):
    # generate the event identifiers of the individual event mentions by
    # chaining nested event arguments
    # single hop represents just the kb link of the arguments
    for task in tasks:
        for arg in ["arg0", "arg1", "argL", "argT"]:
            task[arg] = task[arg].split("/")

    for task in tasks:
        rs_id = task["roleset_id"]
        if rs_id in syn_map:
            rs_id = syn_map[rs_id]
        task["EID_single"] = list(
            itertools.product([task["topic"]], task["arg0"], [rs_id], task["arg1"])
        )
        task["EID_lt"] = list(
            itertools.product(
                [task["topic"]], task["arg0"], [rs_id], task["argL"], task["argT"]
            )
        )

    sent_rs_id2task = {}
    for task in tqdm(tasks, desc="generating eids"):
        sent_rs_id = (task["doc_id"], task["sentence_id"], task["roleset_id"])
        if sent_rs_id not in sent_rs_id2task:
            sent_rs_id2task[sent_rs_id] = []
        sent_rs_id2task[sent_rs_id].append(task)

    for task in tasks:
        add_eid(task, syn_map, sent_rs_id2task)


IMP_KEYS_COR = [
    "arg0",
    "arg1",
    "argL",
    "argT",
    "roleset_id",
    "lemma",
    "doc_id",
    "sentence_id",
    "mention_id",
    "gold_cluster",
    "topic",
]


def clean_task(task):
    return {k: task[k] for k in IMP_KEYS_COR}


def resolve_dict(key2cluster_arr):
    key2cluster = {}
    all_keys = sorted(list(key2cluster_arr.keys()))
    key2i = {k: i for i, k in enumerate(all_keys)}
    n = len(all_keys)
    cluster2keys = {}
    for key, cluster in key2cluster_arr.items():
        for c in cluster:
            if c not in cluster2keys:
                cluster2keys[c] = []
            cluster2keys[c].append(key)

    key_mat = lil_matrix((n, n))

    for cluster in cluster2keys.values():
        for k1 in cluster:
            for k2 in cluster:
                key_mat[key2i[k1], key2i[k2]] = 1

    adj_matrix = key_mat.tocsr()

    # Find connected components
    n_components, labels = connected_components(
        csgraph=adj_matrix, directed=False, return_labels=True
    )
    for k, l in zip(all_keys, labels):
        key2cluster[k] = l
    return key2cluster


def get_syn_map_vn(roleset_dict_path):
    import pickle

    pb_dict = pickle.load(open(roleset_dict_path, "rb"))
    rs2cluster_arr = {}
    for rs, rs_dict in pb_dict.items():
        if rs not in rs2cluster_arr:
            rs2cluster_arr[rs] = []
            for lexlink in rs_dict["lexlinks"]:
                if lexlink["resource"] == "VerbNet":
                    rs2cluster_arr[rs].append(lexlink["class"][0].split(".")[0])
    syn_vn = resolve_dict(rs2cluster_arr)
    return syn_vn


# ---------------------- Commands ------------------------- #


@app.command()
def dummy():
    pass


@app.command()
def run_stupidly_large_exp(tasks_file_path):
    big_tasks_a = []
    bit_tasks_b = []
    tasks_a = list([clean_task(t) for t in load_tasks(tasks_file_path)])
    for i in range(100):
        tasks__a = copy.deepcopy(tasks_a)
        for t in tasks__a:
            t["mention_id"] += "_" + str(i)
            t["doc_id"] += "_" + str(i)
            t["sentence_id"] += "_" + str(i)
            # t['topic'] += '_' + str(i)
        big_tasks_a.extend(tasks__a)
        tasks__b = copy.deepcopy(tasks__a)
        bit_tasks_b.extend(tasks__b)

    print(len(big_tasks_a))

    start_time = timeit.default_timer()
    generate_eids(big_tasks_a, {})
    eid_N_a = [(t["mention_id"], t["EIDs"]) for t in big_tasks_a]
    generate_eids(bit_tasks_b, {})
    eid_N_b = [(t["mention_id"], t["EIDs"]) for t in bit_tasks_b]

    print("clustering")
    eid_N_a_and_b = and_clustering(eid_N_a, eid_N_b)
    elapsed = timeit.default_timer() - start_time

    print("total elapsed time for the experiment", elapsed)


@app.command()
def and_ann_results(a1_path, a2_path, use_vn: Optional[bool] = False):
    a1_tasks = list(load_tasks(a1_path))
    a2_tasks = list(load_tasks(a2_path))

    print(len(a1_tasks))
    print(len(a2_tasks))

    pb_syn_map = get_syn_map_vn("./outputs/common/pb.dict")
    if not use_vn:
        pb_syn_map = {}

    generate_eids(a1_tasks, pb_syn_map)
    generate_eids(a2_tasks, pb_syn_map)
    gold_clusters = [(t["mention_id"], t["gold_cluster"]) for t in a1_tasks]

    eid_N_1 = [(t["mention_id"], t["EIDs"]) for t in a1_tasks]
    eid_N_2 = [(t["mention_id"], t["EIDs"]) for t in a2_tasks]

    eid_N_1and2 = and_clustering(eid_N_1, eid_N_2)
    eid_N = [(m_id, (str(c) + "_N",)) for m_id, c in eid_N_1and2]
    run_coreference_results(gold_clusters, eid_N_1and2, "& \\eidNH")

    eid_lts_1 = [(t["mention_id"], t["EID_lt"]) for t in a1_tasks]
    eid_lts_2 = [(t["mention_id"], t["EID_lt"]) for t in a2_tasks]

    eid_lt_clus = and_clustering(eid_lts_1, eid_lts_2)
    eid_lts = [(m_id, (str(c) + "_LT",)) for m_id, c in eid_lt_clus]
    run_coreference_results(gold_clusters, eid_lt_clus, "& \\eidLTH")

    eid_N_clus_and_lt = and_clustering(eid_N, eid_lts)
    run_coreference_results(
        gold_clusters, eid_N_clus_and_lt, "& \\eidNH~\\AND~\\eidLTH"
    )

    eid_N_clus_or_lt = or_clustering(eid_N, eid_lts)
    run_coreference_results(gold_clusters, eid_N_clus_or_lt, "& \\eidNH~\\OR~\\eidLTH")


@app.command()
def or_ann_results(a1_path, a2_path, use_vn: Optional[bool] = False):
    a1_tasks = list(load_tasks(a1_path))
    a2_tasks = list(load_tasks(a2_path))

    print(len(a1_tasks))
    print(len(a2_tasks))

    pb_syn_map = get_syn_map_vn("./outputs/common/pb.dict")
    if not use_vn:
        pb_syn_map = {}

    generate_eids(a1_tasks, pb_syn_map)
    generate_eids(a2_tasks, pb_syn_map)
    gold_clusters = [(t["mention_id"], t["gold_cluster"]) for t in a1_tasks]

    eid_N_1 = [(t["mention_id"], t["EIDs"]) for t in a1_tasks]
    eid_N_2 = [(t["mention_id"], t["EIDs"]) for t in a2_tasks]

    eid_N_1or2 = or_clustering(eid_N_1, eid_N_2)
    eid_N = [(m_id, (str(c) + "_N",)) for m_id, c in eid_N_1or2]
    run_coreference_results(gold_clusters, eid_N_1or2, "& \\eidNH")

    eid_lts_1 = [(t["mention_id"], t["EID_lt"]) for t in a1_tasks]
    eid_lts_2 = [(t["mention_id"], t["EID_lt"]) for t in a2_tasks]

    eid_lt_clus = or_clustering(eid_lts_1, eid_lts_2)
    eid_lts = [(m_id, (str(c) + "_LT",)) for m_id, c in eid_lt_clus]
    run_coreference_results(gold_clusters, eid_lt_clus, "& \\eidLTH")

    eid_N_clus_and_lt = and_clustering(eid_N, eid_lts)
    run_coreference_results(
        gold_clusters, eid_N_clus_and_lt, "& \\eidNH~\\AND~\\eidLTH"
    )

    eid_N_clus_or_lt = or_clustering(eid_N, eid_lts)
    run_coreference_results(gold_clusters, eid_N_clus_or_lt, "& \\eidNH~\\OR~\\eidLTH")


@app.command()
def single_ann_results(tasks_file_path, use_vn: Optional[bool] = False):
    if str(tasks_file_path).endswith("jsonl"):
        tasks = list(load_tasks(tasks_file_path))
    else:
        tasks = list(load_tasks(tasks_file_path))

    tasks = [{k: t[k] for k in IMP_KEYS_COR} for t in tasks]

    pb_syn_map = {}
    if use_vn:
        pb_syn_map = get_syn_map_vn("./outputs/common/pb.dict")
    print(len(pb_syn_map))
    print(len(tasks))
    generate_eids(tasks, pb_syn_map)

    gold_clusters = [(t["mention_id"], t["gold_cluster"]) for t in tasks]
    # Lemma-Only
    pred_clusters = [(t["mention_id"], t["lemma"]) for t in tasks]
    run_coreference_results(gold_clusters, pred_clusters, "\\LEM")

    # RS-ONLY
    pred_clusters = [(t["mention_id"], t["roleset_id"]) for t in tasks]
    run_coreference_results(gold_clusters, pred_clusters, "& \\RSHum")

    if pb_syn_map and len(pb_syn_map):
        pred_clusters = [
            (
                t["mention_id"],
                pb_syn_map[t["roleset_id"]]
                if t["roleset_id"] in pb_syn_map
                else t["roleset_id"],
            )
            for t in tasks
        ]
        run_coreference_results(gold_clusters, pred_clusters, "& \\PBHVN")

    # EID-0-hop
    eid_0s = [(t["mention_id"], t["EID_single"]) for t in tasks]
    eid_0_clus = a_clustering(eid_0s)
    # run_coreference_results(gold_clusters, eid_0_clus, 'EID-0-hop')

    # eid-0 and eid_lt
    # eid_0_and_lt = and_clustering(eid_0s, eid_lts)
    # run_coreference_results(gold_clusters, eid_0_and_lt, 'EID_0_and_lt')

    # eid-0 or eid_lt
    # eid_0_or_lt = or_clustering(eid_0s, eid_lts)
    # run_coreference_results(gold_clusters, eid_0_or_lt, 'EID_0_or_lt')

    # eid-N
    # generate_eids(tasks)
    eid_N = [(t["mention_id"], t["EIDs"]) for t in tasks]
    eid_N_clus = a_clustering(eid_N)
    run_coreference_results(gold_clusters, eid_N_clus, "& \\eidNH")

    eid_lts = [(t["mention_id"], t["EID_lt"]) for t in tasks]
    eid_lt_clus = a_clustering(eid_lts)
    run_coreference_results(gold_clusters, eid_lt_clus, "& \\eidLTH")

    eid_N_clus_and_lt = and_clustering(eid_N, eid_lts)
    run_coreference_results(
        gold_clusters, eid_N_clus_and_lt, "& \\eidNH~\\AND~\\eidLTH"
    )

    eid_N_clus_or_lt = or_clustering(eid_N, eid_lts)
    run_coreference_results(gold_clusters, eid_N_clus_or_lt, "& \\eidNH~\\OR~\\eidLTH")

    # eid_N_clus_or_0 = and_clustering(eid_N, eid_0s)
    # run_coreference_results(gold_clusters, eid_N_clus_or_0, 'eid_N_and_0')


@app.command()
def gpt_results(tasks_file_path, use_vn: Optional[bool] = False):
    tasks = list(load_tasks(tasks_file_path))
    gold_clusters = [(t["mention_id"], t["gold_cluster"]) for t in tasks]
    # Lemma-Only
    pred_clusters = [(t["mention_id"], t["roleset_id"]) for t in tasks]
    run_coreference_results(gold_clusters, pred_clusters, "\\RSGPT")

    generate_eids(tasks, {})

    eid_N = [(t["mention_id"], t["EIDs"]) for t in tasks]
    eid_N_clus = a_clustering(eid_N)
    run_coreference_results(gold_clusters, eid_N_clus, "& \\eidNH")

    eid_lts = [(t["mention_id"], t["EID_lt"]) for t in tasks]
    eid_lt_clus = a_clustering(eid_lts)
    run_coreference_results(gold_clusters, eid_lt_clus, "& \\eidLTH")

    eid_N_clus_and_lt = and_clustering(eid_N, eid_lts)
    run_coreference_results(
        gold_clusters, eid_N_clus_and_lt, "& \\eidNH~\\AND~\\eidLTH"
    )

    eid_N_clus_or_lt = or_clustering(eid_N, eid_lts)
    run_coreference_results(gold_clusters, eid_N_clus_or_lt, "& \\eidNH~\\OR~\\eidLTH")


if __name__ == "__main__":
    app()
