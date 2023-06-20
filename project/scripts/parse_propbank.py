from spacy.util import ensure_path
from bs4 import BeautifulSoup
from pathlib import Path
from glob import glob
from tqdm import tqdm
import pickle
import typer

app = typer.Typer()
# ------------------------ Helpers  ------------------------ #


def _get_eg_arg_spans(eg_):
    eg_text = eg_.find('text').text
    if eg_.find('propbank') and \
            eg_.find('propbank').find('rel') and \
            eg_.find('propbank').find('rel')['relloc'] != '?':
        t_indices = [int(i) for i in eg_.find('propbank').find('rel')['relloc'].split()]
        t_start = t_indices[0]
        t_end = t_indices[-1]

        if t_start > t_end:
            return None

        args = eg_.find('propbank').find_all('arg')
        arg_token_map = []
        for a in args:
            try:
                a_map = {'label': a['type'].replace('ARG', 'ARG-'),
                         'token_start': int(a['start']), 'token_end': int(a['end'])}
                if int(a['start']) <= int(a['end']):
                    arg_token_map.append(a_map)
            except ValueError:
                pass

        return {
            'src': eg_['src'].replace('ontonotes ', 'ontonotes/'),
            'text': eg_text,
            'head_span': {
                'label': 'V',
                'token_start': t_start,
                'token_end': t_end,
            },
            'relations': arg_token_map
        }
    else:
        return None


# ------------------------ Commands ------------------------ #
@app.command()
def save_pb_dict(
        input_path: Path,
        output_path: Path
):
    input_path = ensure_path(input_path)
    output_path = ensure_path(output_path)

    if not input_path.exists():
        raise ValueError(f"Could not find {input_path}.")
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    frame_files = list(glob(str(input_path) + '/*.xml'))
    pb_dict = {}
    for frame in tqdm(frame_files, desc='Reading FrameSet'):
        with open(frame) as ff:
            frame_bs = BeautifulSoup(ff.read(), parser='lxml', features="lxml")
            predicate = frame_bs.find('predicate')['lemma']
            rolesets = frame_bs.find_all('roleset')
            for roleset in rolesets:
                rs_id = roleset['id']
                rs_defs = {
                    'sense': rs_id,
                    'lemma': predicate,
                    'definition': roleset['name'],
                    'roles': [
                        {'id': 'ARG-' + r['n'], 'definition': r['descr']}
                        for r in roleset.find('roles').find_all('role')
                    ],
                }
                rs_examples = [_get_eg_arg_spans(eg_) for eg_ in roleset.find_all('example')]
                rs_defs['examples'] = rs_examples
                pb_dict[rs_id] = rs_defs

    pickle.dump(pb_dict, open(output_path, 'wb'))


if __name__ == '__main__':
    app()
