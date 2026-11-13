"""Apply a compact regression forest alongside a specified official B or C checkpoint."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def save_forest(path, forest, checkpoint, mode, blend, target_scale):
    arrays = {k: [] for k in ('feature','threshold','left','right','value')}
    roots = []
    offset = 0
    for estimator in forest.estimators_:
        tree = estimator.tree_
        roots.append(offset)
        arrays['feature'].append(tree.feature)
        arrays['threshold'].append(tree.threshold)
        arrays['left'].append(tree.children_left + offset)
        arrays['right'].append(tree.children_right + offset)
        arrays['value'].append(tree.value[:, 0, 0])
        offset += tree.node_count
    np.savez_compressed(path, **{k: np.concatenate(v) for k,v in arrays.items()},
                        roots=roots,mode=mode,blend=blend,target_scale=target_scale,
                        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest())


class CorrectedModel:
    def __init__(self, model, checkpoint, correction):
        self.model = model
        self.meta = model.meta
        with np.load(correction, allow_pickle=False) as data:
            if str(data['digest']) != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
                raise ValueError('Correction belongs to a different base checkpoint')
            self.tree = {k: data[k] for k in data.files}

    def predict(self,x):
        # sklearn trains and routes on float32 feature values.
        features = x.astype(np.float32)
        base = self.model.predict(x)
        t = self.tree
        predictions = np.zeros(len(x))
        for root in t['roots']:
            nodes = np.full(len(x),root)
            while True:
                rows = np.flatnonzero(t['feature'][nodes]>=0)
                if not len(rows):
                    break
                parents = nodes[rows]
                left = features[rows,t['feature'][parents]] <= t['threshold'][parents]
                nodes[rows] = np.where(left,t['left'][parents],t['right'][parents])
            predictions += t['value'][nodes]
        predictions *= float(t['target_scale']) / len(t['roots'])
        delta = predictions if str(t['mode'])=='residual' else predictions-base
        return np.maximum(0,base+float(t['blend'])*delta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--correction',type=Path,required=True)
    args = parser.parse_args()
    sys.path.insert(0,str(args.package.resolve()))
    from baselines.model import Model
    from baselines.predict import predict
    model = CorrectedModel(Model.load(args.checkpoint),args.checkpoint,args.correction)
    for line in sys.stdin:
        if line.strip():
            row = json.loads(line)
            if set(row)-{'inputs','task'} or row.get('task','regression')!='regression':
                raise ValueError('Expected an input-only regression record')
            print(json.dumps(predict(model,row['inputs']),allow_nan=False),flush=True)


if __name__=='__main__':
    main()
