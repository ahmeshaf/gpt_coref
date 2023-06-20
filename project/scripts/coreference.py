from coval.coval.conll.reader import get_coref_infos
from coval.coval.eval.evaluator import evaluate_documents as evaluate
from coval.coval.eval.evaluator import muc, b_cubed, ceafe, lea
from prodigy.components.loaders import JSONL, JSON
import numpy as np
import os


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
    with open(out_file_path, 'w') as of:
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
    gold_key_file = f'evt_gold.keyfile'
    generate_key_file(gold_clusters, 'evt', './', gold_key_file)
    system_key_file = './evt_annotated.keyfile'
    generate_key_file(predicted_clusters, 'evt', './', system_key_file)

    def read(key, response):
        return get_coref_infos('%s' % key, '%s' % response,
                               False, False, True)

    doc = read(gold_key_file, system_key_file)
    mr, mp, mf = np.round(np.round(evaluate(doc, muc), 3) * 100, 1)
    br, bp, bf = np.round(np.round(evaluate(doc, b_cubed), 3) * 100, 1)
    cp, cr, cf = np.round(np.round(evaluate(doc, ceafe), 3) * 100, 1)
    lr, lp, lf = np.round(np.round(evaluate(doc, lea), 3) * 100, 1)

    recll = np.round((mr + br)/2, 1)
    precision = np.round((mp + bp)/2, 1)
    connl = np.round((mf + bf + cf) / 3, 1)

    print(f'{method_name} &&', recll, '&', precision, '&', connl, '&', lf, '\\\\')


def single_ann_results(tasks_file_path):
    tasks = list(JSONL(tasks_file_path))

    gold_clusters = [(t['mention_id'], t['gold_cluster']) for t in tasks]
    # Lemma-Only
    pred_clusters = [(t['mention_id'], t['lemma']) for t in tasks]
    run_coreference_results(gold_clusters, pred_clusters, 'Lemma-Only')

    # RS-ONLY
    pred_clusters = [(t['mention_id'], t['roleset_id']) for t in tasks]
    run_coreference_results(gold_clusters, pred_clusters, 'RS-Only')


if __name__ == '__main__':
    dev_ann1 = '../annotations/ecb/dev/dev_full_rs_ann1.jsonl'
    single_ann_results(dev_ann1)
