"""Local candidate for contest_harness 1.0.2 workspace-v2 rounds.

One invocation performs one hosted round. The host owns Qwen credentials,
measurement and metering; this module only writes in /workspace.
"""
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
from baselines.model import Model
from baselines.data import read_dataset
from b_bagging_adam import migrate, fit_teacher, fit_readout_adam

TASKS = ('classification', 'regression')
SCHEDULES = (
    'def lr_factor(epoch):\n    return 1.0\n',
    'def lr_factor(epoch):\n    return 0.98 ** epoch\n',
    'def lr_factor(epoch):\n    return 0.95 ** epoch\n',
    'def lr_factor(epoch):\n    return 0.9 ** epoch\n',
    'def lr_factor(epoch):\n    return 1.0 / (1.0 + 0.05 * epoch)\n',
)
LRS = (0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01)
EPOCHS = (3, 5, 8)
ID_RE = re.compile(r'^xjtu_Bearing[1-3]_[1-5]_s[0-9]+$')


def _sha(path):
    return 'sha256:' + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_projection(rows, path):
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps({k: row[k] for k in ('id', 'inputs', 'answer')}, ensure_ascii=False) + '\n')


def _parse(text, source):
    match = re.search(r'\{.*\}', text, re.S)
    if match is None:
        raise ValueError('Model returned no JSON decision')
    action = json.loads(match.group())
    lr, ep = action.get('learning_rate'), action.get('epochs')
    if isinstance(lr, bool) or lr not in LRS or type(ep) is not int or ep not in EPOCHS:
        raise ValueError('Model decision outside reviewed training bounds')
    edits = action.get('edits')
    if not isinstance(edits, list) or len(edits) > 1:
        raise ValueError('Expected at most one precise schedule edit')
    new_source = source
    if edits:
        edit = edits[0]
        if not isinstance(edit, dict) or edit.get('find', '').strip() != source.strip():
            raise ValueError('Schedule edit did not match current code')
        new_source = str(edit.get('replace', '')).strip() + '\n'
        if new_source not in SCHEDULES:
            raise ValueError('Unreviewed executable code rejected')
    if not isinstance(action.get('reason'), str) or not action['reason'].strip():
        raise ValueError('Model gave no reasoning')
    return action, new_source


def _metric(reply, task):
    if not isinstance(reply, dict):
        return None
    for obj in (reply, reply.get('summary'), reply.get('validation'), reply.get('metrics')):
        if isinstance(obj, dict):
            metrics = obj.get('metrics', obj)
            if isinstance(metrics, dict):
                name = 'macro_f1' if task == 'classification' else 'rmse'
                if isinstance(metrics.get(name), (int, float)) and np.isfinite(metrics[name]):
                    return metrics
    return None


def _accept(parent, candidate, task, group_guard):
    current, proposed = _metric(parent, task), _metric(candidate, task)
    if proposed is None:
        return False, 'candidate platform measurement missing'
    if current is None:
        return False, 'parent platform measurement missing'
    primary = 'macro_f1' if task == 'classification' else 'rmse'
    improved = proposed[primary] > current[primary] if task == 'classification' else proposed[primary] < current[primary]
    if group_guard and improved:
        a, b = current.get('group_mean_rmse'), proposed.get('group_mean_rmse')
        improved = isinstance(a, (int, float)) and isinstance(b, (int, float)) and b < a
    return improved, 'host measurement improved' if improved else 'host measurement did not meet gate'


def train(rnd):
    if rnd.dataset_key in ('A', 'C'):
        from ac_submission import train as train_ac
        return train_ac(rnd)
    if rnd.task.get('task') not in TASKS:
        raise ValueError('Unsupported task')
    work = Path(os.environ.get('CONTEST_WORK', '/workspace'))
    work.mkdir(parents=True, exist_ok=True)
    original_path = rnd.parent.path('checkpoint.npz')
    parent = Model.load(original_path)
    if rnd.dataset_key != parent.meta.get('profile') or rnd.task['task'] != parent.meta['task']:
        raise ValueError('Parent task or profile mismatch')
    history = rnd.history()
    training_path, validation_path = work / 'train.jsonl', work / 'validation.jsonl'
    _read_projection(rnd.data.training_rows(), training_path)
    _read_projection(rnd.data.validation_rows(), validation_path)
    train_set = read_dataset(training_path, parent.meta['task'], parent.meta['classes'], parent.meta['paths'])
    validation_set = read_dataset(validation_path, parent.meta['task'], parent.meta['classes'], parent.meta['paths'])
    if train_set['paths'] != parent.meta['paths'] or validation_set['paths'] != parent.meta['paths']:
        raise ValueError('Feature schema changed')

    state = Path(rnd.state_path) if rnd.state_path else None
    state_info = None
    if state is not None and (state / 'policy.json').is_file():
        state_info = json.loads((state / 'policy.json').read_text(encoding='utf-8'))
    source = state_info.get('source', SCHEDULES[0]) if state_info else SCHEDULES[0]
    if source not in SCHEDULES:
        raise ValueError('Previous schedule source not reviewed')
    valid_group = (rnd.dataset_key == 'B' and all(ID_RE.fullmatch(s) for s in train_set['ids'])
                   and 2 <= len({s.rsplit('_s', 1)[0] for s in train_set['ids']}) <= 30)
    teacher = None
    strategy = 'native_adam'
    if rnd.dataset_key == 'B' and valid_group:
        strategy = 'train_only_bearing_teacher_adam_readout'
        if parent.meta['hidden'] == 24:
            student = migrate(parent, train_set['x'])
        elif parent.meta['hidden'] == 128:
            student = parent
        else:
            raise ValueError('B architecture incompatible')
        if state is not None and (state / 'teacher.npz').is_file():
            if state_info is None or state_info.get('teacher_sha256') != _sha(state / 'teacher.npz'):
                raise ValueError('Teacher state digest mismatch')
            if state_info.get('train_sha256') != train_set['sha256']:
                raise ValueError('Training projection changed; teacher invalid')
            teacher = Model.load(state / 'teacher.npz')
        elif parent.meta['hidden'] == 24:
            teacher = fit_teacher(student, train_set['x'], train_set['y'], train_set['ids'])
        else:
            raise ValueError('128-unit B parent missing its original teacher state')
        if not np.array_equal(student.mean, teacher.mean) or not np.array_equal(student.scale, teacher.scale):
            raise ValueError('Teacher and student normalization differ')
        if not np.array_equal(student.params['w1'], teacher.params['w1']):
            raise ValueError('Teacher and student hidden features differ')
    else:
        if rnd.dataset_key == 'B' and parent.meta['hidden'] == 128 and state is None:
            raise ValueError('128-unit B parent missing teacher state')
        student = parent
        if rnd.dataset_key == 'B':
            strategy = 'anonymous_ids_native_adam_fallback'
    if state_info and state_info.get('strategy') != strategy:
        raise ValueError('Training strategy changed during continuation')

    primary = 'macro_f1' if rnd.task['task'] == 'classification' else 'rmse'
    def compact(value):
        measured = _metric(value, rnd.task['task']) or {}
        return {k: measured[k] for k in (primary, 'group_mean_rmse') if k in measured}
    card = {'dataset': rnd.dataset_key, 'round': rnd.round_index, 'strategy': strategy,
            'parent_sha256': rnd.parent.checkpoint_sha256,
            'history': [{'id': item['id'], 'round_index': item.get('round_index'),
                         'selected_artifact_id': item.get('selected_artifact_id'),
                         'measurement': compact(item.get('summary'))} for item in history[-5:]],
            'parent_measurement': compact(rnd.parent.receipt),
            'current_source': source, 'allowed_sources': list(SCHEDULES),
            'allowed_learning_rate': list(LRS), 'allowed_epochs': list(EPOCHS),
            'train_code': 'Adam readout distillation' if teacher is not None else 'official Model.fit Adam',
            'teacher_only_from_public_train': teacher is not None,
            'anonymous_bearing_ids': rnd.dataset_key == 'B' and not valid_group}
    response = rnd.model([
        {'role': 'system', 'content': 'Read history and training-code state. Make one real code/parameter experiment. '
         'Return JSON only: {"edits":[{"find":"exact current_source","replace":"one allowed_source"}],'
         '"learning_rate":0.003,"epochs":5,"reason":"brief evidence-based reason"}. '
         'Do not repeat the official host prompt. The candidate is selected using host measurements.'},
        {'role': 'user', 'content': json.dumps(card, ensure_ascii=False)}],
        max_tokens=min(640, rnd.max_output_tokens), temperature=0)
    action, new_source = _parse(response.text, source)
    candidate = work / 'candidate'
    candidate.mkdir(exist_ok=False)
    (candidate / 'train_policy.py').write_text(new_source, encoding='utf-8')
    # The only executable schedule sources are literal-reviewed functions.
    namespace = {}
    exec(compile(new_source, 'train_policy.py', 'exec'), {}, namespace)
    schedule = namespace['lr_factor']
    before_steps = student.meta['steps']
    losses = []
    if teacher is None:
        student.meta['learning_rate'] = float(action['learning_rate'])
        for epoch in range(action['epochs']):
            student.meta['learning_rate'] = action['learning_rate'] * schedule(epoch)
            losses.append(float(student.fit(train_set['x'], train_set['y'], 1)[0]))
    else:
        target = teacher.forward(teacher.normalized(train_set['x']))[1][:, 0]
        teacher_file = work / 'teacher.npz'
        teacher.save(teacher_file)
        student.meta['readout_distillation'] = {'optimizer': 'Adam_readout_only',
            'teacher_sha256': _sha(teacher_file),
            'train_sha256': train_set['sha256'], 'frozen': ['w1', 'b1']}
        for epoch in range(action['epochs']):
            lr = action['learning_rate'] * schedule(epoch)
            losses.append(float(fit_readout_adam(student, train_set['x'], target, 1, lr)[0]))
    expected = before_steps + action['epochs'] * int(np.ceil(len(train_set['x']) / student.meta['batch_size']))
    if student.meta['steps'] != expected:
        raise AssertionError('Adam step count mismatch')
    student.save(candidate / 'checkpoint.npz')
    note = {'profile': rnd.dataset_key, 'round': rnd.round_index, 'strategy': strategy,
            'parent_sha256': rnd.parent.checkpoint_sha256, 'checkpoint_sha256': _sha(candidate / 'checkpoint.npz'),
            'initial_steps': before_steps, 'final_steps': expected,
            'learning_rate': action['learning_rate'], 'epochs': action['epochs'],
            'source_sha256': _sha(candidate / 'train_policy.py'), 'train_sha256': train_set['sha256'],
            'validation_sha256': validation_set['sha256'], 'teacher_sha256': None,
            'code_edited': new_source != source, 'training_losses': losses,
            'qwen_call_id': response.call_id, 'reason': action['reason']}
    if teacher is not None:
        note['teacher_sha256'] = _sha(teacher_file)
    (candidate / 'training.json').write_text(json.dumps(note, ensure_ascii=False, indent=2), encoding='utf-8')
    hypothesis = str(action['reason'])[:4000]
    submitted = rnd.submit_candidate(candidate, hypothesis=hypothesis,
        change_summary=f'{strategy}; Adam {action["epochs"]} epochs, steps {before_steps}->{expected}; schedule edit={new_source != source}',
        configuration={'strategy': strategy, 'learning_rate': action['learning_rate'],
                       'epochs': action['epochs'], 'source_sha256': note['source_sha256'],
                       'parent_sha256': rnd.parent.checkpoint_sha256},
        seed=student.meta['seed'], checkpoint_paths=['checkpoint.npz'],
        code_paths=['train_policy.py', 'training.json'],
        predict_argv=['python3', '/app/harness.py', 'predict'])
    accepted, gate = _accept(rnd.parent.receipt, submitted, rnd.task['task'], teacher is not None)
    selected = submitted['artifact_id'] if accepted else rnd.parent.artifact_id
    state_out = rnd.state_workspace(work / 'state')
    if teacher is not None:
        teacher_out = state_out / 'teacher.npz'
        if not teacher_out.exists():
            teacher.save(teacher_out)
        if _sha(teacher_out) != note['teacher_sha256']:
            raise ValueError('Teacher changed between training and state saving')
    policy = {'source': new_source if accepted else source, 'strategy': strategy}
    if teacher is not None:
        policy.update(teacher_sha256=note['teacher_sha256'], train_sha256=train_set['sha256'])
    (state_out / 'policy.json').write_text(json.dumps(policy), encoding='utf-8')
    rnd.save_state(state_out)
    rnd.finish(selected_artifact_id=selected, hypothesis=hypothesis,
        change_summary=f'{strategy}; candidate measured by host; {gate}',
        reason=f'{action["reason"]}; {gate}; model call {response.call_id}',
        configuration={'strategy': strategy, 'candidate_sha256': note['checkpoint_sha256'],
                       'parent_sha256': rnd.parent.checkpoint_sha256, 'accepted': accepted,
                       'qwen_call_id': response.call_id, 'host_measurement': compact(submitted)},
        seed=student.meta['seed'], history_ids=[item['id'] for item in history],
        model_call_ids=list(rnd.call_ids))


_prediction_models = {}


def predict(sample, model_dir):
    path = Path(os.environ.get('CONTEST_ARTIFACT_PATH', Path(model_dir) / 'checkpoint.npz'))
    # /model is immutable for the lifetime of a prediction process.
    key = str(path)
    if key not in _prediction_models:
        _prediction_models[key] = Model.load(path)
    model = _prediction_models[key]
    if sample.task != model.meta['task']:
        raise ValueError('Prediction task mismatch')
    from baselines.data import flatten
    paths, values = flatten(sample.inputs)
    if paths != model.meta['paths']:
        raise ValueError('Prediction feature mismatch')
    prediction = model.predict(np.array([values], dtype=np.float64))[0]
    if sample.task == 'regression':
        return {'value': float(prediction)}
    return [{'label': label, 'score': float(score)} for label, score in zip(model.meta['classes'], prediction)]
