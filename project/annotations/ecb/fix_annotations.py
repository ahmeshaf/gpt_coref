import glob
import json
import pickle
import sys


if __name__ == "__main__":
    anno_dir = sys.argv[1]
    mention_map_file = sys.argv[2]

    mention_map = pickle.load(open(mention_map_file, 'rb'))

    for file_path in glob.glob(anno_dir + "/*/*.json"):
        annos = json.load(open(file_path))

        for task in annos:
            task['marked_sentence'] = mention_map[task['mention_id']]['marked_sentence']
            task['marked_doc'] = mention_map[task['mention_id']]['marked_doc']

            if 'bert_doc' in task:
                task.pop('bert_doc')

        json.dump(annos, open(file_path, 'w'), indent=1)

