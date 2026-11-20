"""A/C full-state continuation through the official SDK."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from baselines.model import Model
from baselines.data import read_dataset
from model_decision import brief_reason, request_decision
import legacy_submission as legacy
import a_stateful_projection as a_method
import c_continuation as c_method

def sha(path):return 'sha256:'+hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_json(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def model_metrics(receipt,task,profile):
    measured=legacy._metric(receipt,task)
    if profile!='C' or measured is None:return measured
    return {key:measured[key] for key in ('rmse','mae','group_mean_rmse') if key in measured}

def parse_action(text, allowed_epochs, allowed_lrs):
    match = re.search(r'\{.*\}', text, re.S)
    if not match:
        raise ValueError('Qwen returned no JSON')
    action = json.loads(match.group())
    if type(action.get('epochs')) is not int or action['epochs'] not in allowed_epochs:
        raise ValueError('Unreviewed epochs')
    if isinstance(action.get('learning_rate'), bool) or action.get('learning_rate') not in allowed_lrs:
        raise ValueError('Unreviewed learning rate')
    if action.get('acceptance_policy') != 'improve_primary':
        raise ValueError('Missing conditional decision')
    return {'epochs': action['epochs'], 'learning_rate': action['learning_rate'],
            'acceptance_policy': action['acceptance_policy'], 'reason': brief_reason(action.get('reason'))}

def train(rnd, parent_validation):
    profile=rnd.dataset_key
    work=Path('/workspace')
    parent_path=rnd.parent.path('checkpoint.npz');parent=Model.load(parent_path)
    if parent.meta['profile']!=profile or parent.meta['task']!=rnd.task['task']:
        raise ValueError('Parent task mismatch')
    for relative in rnd.parent.declared_paths():
        if relative.startswith('training_state/'):rnd.parent.path(relative)
    history=rnd.history()
    training_path=work/'train.jsonl'
    legacy._read_projection(rnd.data.training_rows(),training_path)
    tr=read_dataset(training_path,parent.meta['task'],parent.meta['classes'],parent.meta['paths'])
    state_out=rnd.state_workspace(work/'state');attempts_path=state_out/'attempts.json'
    attempts=json.loads(attempts_path.read_text(encoding='utf8')) if attempts_path.exists() else []
    allowed_epochs=[90,180,270,450,900] if profile=='A' else [10,20,40,60,100]
    allowed_lrs=[.001,.002,.003] if profile=='A' else [.0003,.001,.003]
    epochs_before=parent.meta.get('a_stateful_projection',{}).get('branch_epochs',0) if profile=='A' else parent.meta.get('c_continuation',{}).get('training_epochs',0)
    strategy='train_only_pair_teacher_stateful_projection' if profile=='A' else 'train_only_condition_noise_stateful_adam'
    card={'profile':profile,'round':rnd.round_index,'strategy':strategy,
          'parent_sha256':rnd.parent.checkpoint_sha256,
          'parent_measurement':model_metrics(parent_validation,rnd.task['task'],profile),
          'training_epochs_already_completed':epochs_before,
          'allowed_epochs':allowed_epochs,'allowed_learning_rate':allowed_lrs,
          'last_experiments':attempts[-8:],
          'training_code':'Exact train-only distance-ratio teacher; 96-unit Adam branch; gain=2 and blend=.5.' if profile=='A' else 'Train-only condition residual transform; 96-unit latent Adam branch; finite noise weight=.1.',
          'checkpoint_semantics':'Prediction NPZ paired with verified complete training_state; resume with this method, not Model.fit on projection.',
          'reference_recipe':{'learning_rate':.003,'total_epochs':900 if profile=='A' else 100},
          'acceptance_policy':'improve_primary'}
    action,response=request_decision(rnd,[
        {'role':'system','content':'Choose one real continuation experiment using allowed epochs and learning rates. Read failed experiments and avoid repeating the same choice from the same parent. Return JSON only: {"epochs":90,"learning_rate":0.003,"acceptance_policy":"improve_primary","reason":"explain"}. This policy means retain parent unless host primary metric strictly improves; this is your explicit conditional selection decision. Epochs are actual training epochs, not checkpoint rounds.'},
        {'role':'user','content':json.dumps(card,ensure_ascii=False)}],
        lambda text: parse_action(text, allowed_epochs, allowed_lrs))
    candidate=work/'candidate';candidate.mkdir();state_dir=candidate/'training_state';begin=time.perf_counter()
    if profile=='A':
        a_method.prepare(parent_path,tr,state_dir)
        note=a_method.train_round(state_dir,action['epochs'],action['learning_rate'])
        a_method.export(state_dir,candidate/'checkpoint.npz')
    else:
        source=parent_path.parent/'training_state'
        if source.exists():shutil.copytree(source,state_dir)
        else:state_dir.mkdir()
        student,native_state=c_method.prepare(parent,tr,state_dir)
        before=student.meta['c_continuation']['branch_steps']
        losses=c_method.train_round(student,tr,state_dir,epochs=action['epochs'],learning_rate=action['learning_rate'],state=native_state)
        student.save(candidate/'checkpoint.npz')
        note={'initial_branch_steps':before,'final_branch_steps':student.meta['c_continuation']['branch_steps'],
              'initial_branch_epochs':epochs_before,'final_branch_epochs':student.meta['c_continuation']['training_epochs'],
              'actual_epochs':action['epochs'],'loss_first':losses[0],'loss_last':losses[-1],
              'prediction_checkpoint_requires_latent_state':True}
    note.update(profile=profile,round=rnd.round_index,strategy=strategy,parent_sha256=rnd.parent.checkpoint_sha256,
                checkpoint_sha256=sha(candidate/'checkpoint.npz'),train_sha256=tr['sha256'],qwen_call_id=response.call_id,
                reason=action['reason'],learning_rate=action['learning_rate'],wall_seconds=time.perf_counter()-begin)
    save_json(candidate/'training.json',note)
    method_file='a_stateful_projection.py' if profile=='A' else 'c_continuation.py'
    for name in ('ac_submission.py',method_file):shutil.copyfile(Path('/app')/name,candidate/name)
    state_files=sorted(p.relative_to(candidate).as_posix() for p in state_dir.rglob('*') if p.is_file())
    submitted=rnd.submit_candidate(candidate,hypothesis=action['reason'],
        change_summary=f'{strategy}; {action["epochs"]} actual epochs; paired optimizer state',
        configuration={'strategy':strategy,**action},seed=44 if profile=='A' else 20261118,
        checkpoint_paths=['checkpoint.npz',*state_files],code_paths=['ac_submission.py',method_file,'training.json'],
        predict_argv=['python3','/app/harness.py','predict'])
    accepted,gate=legacy._accept(parent_validation,submitted,rnd.task['task'],False)
    # Optimizer stays in its artifact; rejected training state cannot affect parent.
    attempts.append({'parent_sha256':rnd.parent.checkpoint_sha256,'epochs':action['epochs'],
                     'learning_rate':action['learning_rate'],'candidate_measurement':model_metrics(submitted,rnd.task['task'],profile),
                     'accepted':accepted,'reason':action['reason']})
    save_json(attempts_path,attempts);rnd.save_state(state_out)
    rnd.finish(selected_artifact_id=submitted['artifact_id'] if accepted else rnd.parent.artifact_id,
        hypothesis=action['reason'],change_summary=f'{strategy}; {gate}',
        reason=f'Qwen specified {action["acceptance_policy"]}; {gate}. {action["reason"]}',
        configuration={'accepted':accepted,'strategy':strategy,'candidate_sha256':note['checkpoint_sha256'],
                       'parent_sha256':rnd.parent.checkpoint_sha256,**action},seed=44 if profile=='A' else 20261118,
        history_ids=[row['id'] for row in history],model_call_ids=list(rnd.call_ids))
