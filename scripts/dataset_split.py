# python scripts/join_seen_data.py /Users/nstanishev/Workspace/epfl/04/dl/project/data/hateful_memes

import os
import sys
import json


def join(filenames, output):
    filenames = [os.path.join(sys.argv[1], f) for f in filenames]
    data = []
    for fname in filenames:
        data.extend([json.loads(jline) for jline in open(fname, 'r').readlines()])
    data = list({d['id']: d for d in data}.values())
    with open(os.path.join(sys.argv[1], output), 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')


join(['dev_seen.jsonl', 'dev_unseen.jsonl'], 'dev.jsonl')
join(['test_seen.jsonl', 'test_unseen.jsonl'], 'test.jsonl')
# join(['train.jsonl', 'dev.jsonl', 'test.jsonl'], 'all.jsonl')
