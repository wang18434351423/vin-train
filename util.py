import json
from tqdm import tqdm


def read_json(path):
    with open(path, encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            dic = json.loads(line)
    return dic
