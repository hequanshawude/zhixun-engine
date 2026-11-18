"""B compatible migration and Adam readout distillation, training arrays only.

The caller loads the original official Model class and supplies the training
arrays. No paths, validation data, cloud clients or platform actions live here.
"""
import copy
import math
import numpy as np


def migrate(base,x):
    """Preserve the original 24 units and every old Adam tensor; add zero outputs."""
    if base.meta['task']!='regression' or base.meta['hidden']!=24 or len(base.meta['paths'])!=34:
        raise ValueError('Expected original B checkpoint: 24 hidden units and 34 features')
    model=copy.deepcopy(base)
    z=model.normalized(x);old=24;extra=104
    w=np.zeros((34,extra));b=np.zeros(extra)
    for j in range(extra):
        i=j%34;q=(.2,.5,.8)[(j//34)%3]
        width=max(float(np.percentile(z[:,i],75)-np.percentile(z[:,i],25)),.02)
        w[i,j]=3/width;b[j]=-np.quantile(z[:,i],q)*w[i,j]
    model.params={
        'w1':np.column_stack([base.params['w1'],w]),
        'b1':np.r_[base.params['b1'],b],
        'w2':np.vstack([base.params['w2'],np.zeros((extra,1))]),
        'b2':base.params['b2'].copy()}
    for name in ('first','second'):
        source=getattr(base,name);dest={k:np.zeros_like(v) for k,v in model.params.items()}
        for key,value in source.items():
            if key=='w1':dest[key][:,:old]=value
            elif key=='b2':dest[key][:]=value
            else:dest[key][:old]=value
        setattr(model,name,dest)
    model.meta.update(hidden=128,parameter_count=sum(v.size for v in model.params.values()))
    model.meta['architecture_migration']={
        'method':'train_quantile_tanh_zero_output','old_hidden':24,'new_hidden':128,
        'old_parameters_preserved':True,'old_adam_moments_preserved':True,
        'new_adam_moments':'zero','normalization':'source_checkpoint_unchanged'}
    model.validate()
    return model


def fit_teacher(migrated,train_x,train_y,train_ids,draws=1024,seed=20261117):
    """Fit the training-only bagging target on a copy, never update the student."""
    if migrated.meta['task']!='regression' or migrated.meta['hidden']!=128:
        raise ValueError('Expected migrated regression model with 128 hidden units')
    if type(draws) is not int or not 1<=draws<=10000:
        raise ValueError('draws must be an integer in 1..10000')
    if len(train_x)!=len(train_y) or len(train_y)!=len(train_ids):
        raise ValueError('Training arrays and identifiers differ in length')
    model=copy.deepcopy(migrated)
    hidden=model.forward(model.normalized(train_x))[0]
    design=np.column_stack([hidden,np.ones(len(train_y))])
    target=(train_y-model.meta['target_mean'])/model.meta['target_scale']
    prior=np.r_[model.params['w2'][:,0],model.params['b2'][0]]
    residual=target-design@prior
    groups=np.array([s.rsplit('_s',1)[0] for s in train_ids]);names=np.unique(groups)
    if len(names)<2:raise ValueError('At least two training bearings are required')
    gram=np.stack([design[groups==g].T@design[groups==g] for g in names])
    rhs=np.stack([design[groups==g].T@residual[groups==g] for g in names])
    counts=np.array([np.sum(groups==g) for g in names])
    penalty=np.eye(129);penalty[-1,-1]=.1
    samples=np.random.default_rng(seed).dirichlet(np.full(len(names),.1),size=draws)
    readouts=[]
    for weights in samples:
        weights=weights*len(train_x)/(weights@counts)
        readouts.append(prior+np.linalg.solve(np.einsum('g,gij->ij',weights,gram)+penalty,weights@rhs))
    mean=np.mean(readouts,axis=0)
    model.params['w2'][:,0]=mean[:-1];model.params['b2'][0]=mean[-1]
    model.meta['teacher_fit']={'method':'bearing_dirichlet_mean_readout','concentration':.1,
        'ridge':1.,'draws':draws,'seed':seed,'purpose':'training_target_only',
        'optimizer_state_valid_for_resume':False}
    model.validate()
    return model


def fit_readout_adam(model,x,target,epochs,lr):
    """Continue genuine Adam on readout weights; freeze hidden tensors and moments.

    target is the teacher's raw normalized output, before nonnegative clipping.
    Epoch-indexed ordering ensures split/save/load/resume is exactly reproducible.
    """
    if type(epochs) is not int or epochs<1:raise ValueError('epochs must be a positive integer')
    if not np.isfinite(lr) or not 0<lr<=1:raise ValueError('learning rate must lie in (0,1]')
    target=np.asarray(target,dtype=np.float64)
    if target.shape!=(len(x),) or not np.all(np.isfinite(target)):
        raise ValueError('A finite raw teacher target is required for each training row')
    hidden=model.forward(model.normalized(x))[0]
    model.meta['learning_rate']=float(lr)
    history=[]
    for _ in range(epochs):
        order=np.random.default_rng(np.random.SeedSequence([model.meta['seed'],model.meta['epochs']])).permutation(len(x))
        total=0.
        for start in range(0,len(order),model.meta['batch_size']):
            ix=order[start:start+model.meta['batch_size']];h=hidden[ix]
            error=(h@model.params['w2']+model.params['b2'])[:,0]-target[ix]
            total+=float(np.sum(error*error))
            delta=(2*error/len(ix))[:,None]
            gradients={'w2':h.T@delta,'b2':delta.sum(0)}
            norm=math.sqrt(sum(float(np.sum(g*g)) for g in gradients.values()))
            model.meta['steps']+=1
            for name,gradient in gradients.items():
                gradient*=min(1.,5./max(norm,1e-12))
                model.first[name]=.9*model.first[name]+.1*gradient
                model.second[name]=.999*model.second[name]+.001*gradient*gradient
                first=model.first[name]/(1-.9**model.meta['steps'])
                second=model.second[name]/(1-.999**model.meta['steps'])
                model.params[name]-=lr*first/(np.sqrt(second)+1e-8)
        model.meta['epochs']+=1;history.append(total/len(x))
    model.validate()
    return history
