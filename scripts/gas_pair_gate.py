"""Fit the gas correction, or predict from an input-only JSONL stream."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/'data/raw/cpu/a'
sys.path.insert(0,str(PKG))
from baselines.model import Model
from baselines.data import read_dataset

def distances(x, prototypes):
    return np.linalg.norm(x[:,None,:]-prototypes[None,:,:],axis=2)

def nearest_mean(d,k):
    return np.partition(d,k-1,axis=1)[:,:k].mean(1)

def fit_confirmation(pos,neg,channel_mask,k,quantile):
    p=pos[:,channel_mask]
    within=distances(p,p)
    np.fill_diagonal(within,np.inf)
    negative=distances(p,neg[:,channel_mask])
    ratios=nearest_mean(within,k)/np.maximum(nearest_mean(negative,k),1e-12)
    return float(np.quantile(ratios,quantile))

def save_confirmed(path,pos,neg,base_digest,variant):
    channels=np.arange(pos.shape[1])
    if variant=='consensus':
        masks=np.array([channels%2==0,channels%2==1,np.ones_like(channels,bool)])
        ks=[1,1,3]
        quantiles=[.9,.9,.9]
    elif variant=='coverage':
        masks=np.array([channels//4!=i for i in range(4)])
        ks=[1]*4
        quantiles=[.95]*4
    else:
        raise ValueError('Unknown gate variant')
    cuts=[fit_confirmation(pos,neg,mask,k,q) for mask,k,q in zip(masks,ks,quantiles)]
    cut=fit_confirmation(pos,neg,np.ones_like(channels,bool),1,.95)
    np.savez(path,pos=pos,neg=neg,base_digest=base_digest,cut=cut,masks=masks,ks=ks,cuts=cuts)

class GasPairGate:
    def __init__(self, checkpoint, gate):
        self.model=Model.load(checkpoint)
        with np.load(gate,allow_pickle=False) as state:
            if str(state['base_digest'])!=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
                raise ValueError('Gate does not match base checkpoint')
            self.pos,self.neg,self.cut=(state[k].copy() for k in ('pos','neg','cut'))
            self.confirmations=list(zip(state['masks'],state['ks'],state['cuts'])) if 'masks' in state else []
        self.e,self.a=(self.model.meta['classes'].index(k) for k in ('Ethanol','Acetaldehyde'))

    @property
    def meta(self):
        return self.model.meta

    def predict(self,x):
        scores=self.model.predict(x)
        pred=scores.argmax(1)
        for start in range(0,len(x),512):
            chunk=x[start:start+512]
            labels=pred[start:start+len(chunk)]
            choose=labels==self.e
            if not choose.any():
                continue
            sub=self.model.normalized(chunk[choose])[:,1::8]
            pos=distances(sub,self.pos).min(1)
            neg=distances(sub,self.neg).min(1)
            passed=pos<self.cut*neg
            for mask,k,cut in self.confirmations:
                pending=np.flatnonzero(passed)
                if not len(pending):
                    break
                subspace=sub[pending][:,mask]
                dpos=nearest_mean(distances(subspace,self.pos[:,mask]),k)
                dneg=nearest_mean(distances(subspace,self.neg[:,mask]),k)
                passed[pending]=dpos<cut*dneg
            hits=start+np.flatnonzero(choose)[passed]
            # Preserve total score and all other class scores. The reassigned
            # top scores implement the gate; they are not calibrated probabilities.
            scores[hits,self.a],scores[hits,self.e]=scores[hits,self.e].copy(),scores[hits,self.a].copy()
        return scores

def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--out',type=Path)
    mode.add_argument('--predict',type=Path,metavar='GATE')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'out/cpu-models/best/a.npz')
    args=parser.parse_args()
    checkpoint=args.checkpoint
    if args.predict:
        from baselines.predict import predict
        gate=GasPairGate(checkpoint,args.predict)
        for line in sys.stdin:
            if line.strip():
                request=json.loads(line)
                if set(request)-{'task','inputs'} or request.get('task','classification')!='classification':
                    raise ValueError('Expected an input-only classification request')
                print(json.dumps(predict(gate,request['inputs']),allow_nan=False),flush=True)
        return
    args.out.mkdir(parents=True,exist_ok=False)
    m=Model.load(checkpoint)
    names=m.meta['classes']
    e,a=(names.index(k) for k in ('Ethanol','Acetaldehyde'))
    def split(s):
        return read_dataset(PKG/f'public/{s}.jsonl','classification',names,m.meta['paths'])
    t=split('train')
    x=m.normalized(t['x'])[:,1::8]
    pos=x[t['y']==a]
    neg=x[t['y']==e]
    within=distances(pos,pos)
    np.fill_diagonal(within,np.inf)
    ratios=within.min(1)/np.maximum(distances(pos,neg).min(1),1e-12)
    v=split('validation')
    y=v['y']
    base=m.predict(v['x']).argmax(1)
    def score(pred):
        cm=np.zeros((len(names),len(names)),int)
        np.add.at(cm,(y,pred),1)
        return dict(macro_f1=float(np.mean(2*cm.diagonal()/(cm.sum(0)+cm.sum(1)))),
                    accuracy=float(np.mean(pred==y)),confusion=cm.tolist(),
                    recovered=int(((base!=a)&(pred==a)&(y==a)).sum()),
                    ethanol_lost=int(((base==e)&(pred==a)&(y==e)).sum()))
    results={'base':score(base),'classes':names,'base_checkpoint':str(checkpoint)}
    for q in (.8,.9,.95):
        path=args.out/f'gate-{q}.npz'
        cut=float(np.quantile(ratios,q))
        np.savez(path,pos=pos,neg=neg,cut=cut,base_digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        gate=GasPairGate(checkpoint,path)
        result=score(gate.predict(v['x']).argmax(1))
        result.update(cut=cut,gate=path.name)
        results[str(q)]=result
    for variant in ('consensus','coverage'):
        path=args.out/f'{variant}.npz'
        save_confirmed(path,pos,neg,hashlib.sha256(checkpoint.read_bytes()).hexdigest(),variant)
        pred=GasPairGate(checkpoint,path).predict(v['x']).argmax(1)
        results[variant]=dict(**score(pred),gate=path.name)
    (args.out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    for k,r in results.items():
        if isinstance(r,dict):
            print(k,{f:v for f,v in r.items() if f!='confusion'},flush=True)

if __name__=='__main__':
    main()
