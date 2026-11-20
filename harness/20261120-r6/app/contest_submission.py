"""Local candidate for contest_harness 1.0.2 workspace-v2 rounds.

One invocation performs one hosted round. The host owns Qwen credentials,
measurement and metering; this module only writes in /workspace.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from baselines.model import Model
from baselines.data import read_dataset
from b_bagging_adam import migrate, fit_teacher, fit_anonymous_teacher, fit_readout_adam
from host_metrics import accept_candidate as _accept, primary_metric, read_metrics as _metric
from model_decision import brief_reason, request_decision

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
B_TEACHER_PATH = 'training_state/b_teacher.npz'


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
    if action.get('acceptance_policy') != 'improve_primary':
        raise ValueError('Missing conditional selection policy')
    edits = action.get('edits')
    if not isinstance(edits, list) or len(edits) > 1:
        raise ValueError('Expected at most one precise schedule edit')
    new_source = source
    if edits:
        edit = edits[0]
        if not isinstance(edit, dict) or not isinstance(edit.get('find'), str) or edit['find'].strip() != source.strip():
            raise ValueError('Schedule edit did not match current code')
        new_source = str(edit.get('replace', '')).strip() + '\n'
        if new_source not in SCHEDULES:
            raise ValueError('Unreviewed executable code rejected')
    action['reason'] = brief_reason(action.get('reason'))
    return action, new_source


def _bearing_groups(ids):
    if not all(ID_RE.fullmatch(sample_id) for sample_id in ids):
        return None
    groups = [sample_id.rsplit('_s', 1)[0] for sample_id in ids]
    return groups if 2 <= len(set(groups)) <= 30 else None


def _anonymous_training_sha256(train):
    rows = np.column_stack((train['x'], train['y']))
    canonical = rows[np.lexsort(rows.T)].astype('<f8', copy=False)
    return 'sha256:' + hashlib.sha256(canonical.tobytes()).hexdigest()


def _group_mean_rmse(model, validation, groups):
    _, membership = np.unique(groups, return_inverse=True)
    errors = model.predict(validation['x']) - validation['y']
    return float(np.sqrt(np.bincount(membership, weights=errors ** 2) / np.bincount(membership)).mean())


def _bearing_gate(parent_measurement, candidate_measurement, parent_path, student, validation):
    before, after = (record.get('group_mean_rmse') for record in (parent_measurement, candidate_measurement))
    source = 'host group mean RMSE'
    if not all(type(value) in (int, float) and 0 <= value < float('inf') for value in (before, after)):
        groups = _bearing_groups(validation['ids'])
        if groups is None:
            return True, 'host RMSE improved and validation groups were unavailable'
        # A continued student can share the parent object. Reload the original artifact for the guard.
        before = _group_mean_rmse(Model.load(parent_path), validation, groups)
        after = _group_mean_rmse(student, validation, groups)
        source = 'public validation group mean RMSE'
    improved = after < before
    return improved, f'host RMSE improved and {source} ' + ('improved' if improved else 'did not improve')


def _compatible_teacher(student, teacher):
    contract = ('task', 'paths', 'hidden', 'target_mean', 'target_scale')
    return (all(student.meta[key] == teacher.meta.get(key) for key in contract)
            and np.array_equal(student.mean, teacher.mean)
            and np.array_equal(student.scale, teacher.scale)
            and all(np.array_equal(student.params[key], teacher.params[key]) for key in ('w1', 'b1')))


def _b_teacher(rnd, student, train, grouped, strategy, train_sha):
    info = student.meta.get('readout_distillation', {})
    if (B_TEACHER_PATH in rnd.parent.declared_paths()
            and info.get('strategy') == strategy and info.get('train_sha256') == train_sha):
        path = rnd.parent.path(B_TEACHER_PATH)
        if _sha(path) == info.get('teacher_sha256'):
            teacher = Model.load(path)
            if _compatible_teacher(student, teacher):
                return teacher, 'selected_parent'
    if grouped:
        teacher = fit_teacher(student, train['x'], train['y'], train['ids'])
    else:
        teacher = fit_anonymous_teacher(student, train['x'], train['y'])
    return teacher, 'refit_from_selected_parent'


def train(rnd):
    task = rnd.task.get('task')
    if task not in TASKS:
        raise ValueError('Unsupported task')
    parent_validation = rnd.frame.get('current_validation')
    if _metric(parent_validation, task) is None:
        parent_validation = rnd.parent.receipt
    workspace = Path(os.environ.get('CONTEST_WORK', '/workspace'))
    workspace.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='zhixun-round-', dir=workspace) as directory:
        work = Path(directory)
        if rnd.dataset_key in ('A', 'C'):
            from ac_submission import train as train_ac
            return train_ac(rnd, parent_validation, work)
        return _train_b(rnd, parent_validation, work)


def _train_b(rnd, parent_validation, work):
    task = rnd.task['task']
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

    source = parent.meta.get('readout_distillation', {}).get('source', SCHEDULES[0])
    if source not in SCHEDULES:
        raise ValueError('Previous schedule source not reviewed')
    valid_group = _bearing_groups(train_set['ids']) is not None
    strategy = ('train_only_bearing_teacher_adam_readout' if valid_group
                else 'train_only_anonymous_ridge_teacher_adam_readout')
    teacher_train_sha = train_set['sha256'] if valid_group else _anonymous_training_sha256(train_set)
    if parent.meta['hidden'] == 24:
        student = migrate(parent, train_set['x'])
    elif parent.meta['hidden'] == 128:
        student = parent
    else:
        raise ValueError('B architecture incompatible')
    teacher, teacher_source = _b_teacher(rnd, student, train_set, valid_group, strategy, teacher_train_sha)

    primary = primary_metric(task)
    def compact(value):
        measured = _metric(value, task) or {}
        return {k: measured[k] for k in (primary, 'group_mean_rmse') if k in measured}

    history_preview = []
    for item in history[-5:]:
        record = item.get('record', item)
        configuration = record.get('configuration') if isinstance(record, dict) else None
        preview = {'id': item['id'], 'round_index': item.get('round_index'),
                   'selected_artifact_id': item.get('selected_artifact_id'),
                   'selected_measurement': compact(item)}
        if isinstance(configuration, dict):
            for key in ('parent_sha256', 'learning_rate', 'epochs', 'accepted', 'acceptance_policy', 'source_sha256'):
                if key in configuration:
                    preview[key] = configuration[key]
            preview['candidate_measurement'] = compact(configuration.get('host_measurement'))
        history_preview.append(preview)
    card = {'dataset': rnd.dataset_key, 'round': rnd.round_index, 'strategy': strategy,
            'parent_sha256': rnd.parent.checkpoint_sha256,
            'history': history_preview,
            'parent_measurement': compact(parent_validation),
            'current_source': source, 'allowed_sources': list(SCHEDULES),
            'allowed_learning_rate': list(LRS), 'allowed_epochs': list(EPOCHS),
            'acceptance_policy': 'improve_primary',
            'train_code': 'Adam readout distillation',
            'teacher_only_from_public_train': True,
            'anonymous_bearing_ids': rnd.dataset_key == 'B' and not valid_group}
    (action, new_source), response = request_decision(rnd, [
        {'role': 'system', 'content': 'Read history and training-code state. Make one real code/parameter experiment. '
         'Return JSON only: {"edits":[{"find":"exact current_source","replace":"one allowed_source"}],'
         '"learning_rate":0.003,"epochs":5,"acceptance_policy":"improve_primary","reason":"brief evidence-based reason"}. '
         'This is your conditional selection: promote only if host RMSE improves and, when applicable and measurable '
         'for named training bearings, validation group mean RMSE also improves. Otherwise retain the parent. '
         'Check prior experiments from the same parent before choosing. Use edits [] to keep the current schedule.'},
        {'role': 'user', 'content': json.dumps(card, ensure_ascii=False)}],
        lambda text: _parse(text, source))
    candidate = work / 'candidate'
    candidate.mkdir(exist_ok=False)
    (candidate / 'train_policy.py').write_text(new_source, encoding='utf-8')
    # The only executable schedule sources are literal-reviewed functions.
    namespace = {}
    exec(compile(new_source, 'train_policy.py', 'exec'), {}, namespace)
    schedule = namespace['lr_factor']
    before_steps = student.meta['steps']
    losses = []
    target = teacher.forward(teacher.normalized(train_set['x']))[1][:, 0]
    teacher_file = candidate / B_TEACHER_PATH
    teacher_file.parent.mkdir()
    teacher.save(teacher_file)
    teacher_sha = _sha(teacher_file)
    student.meta['readout_distillation'] = {'optimizer': 'Adam_readout_only',
        'teacher_sha256': teacher_sha, 'train_sha256': teacher_train_sha,
        'source': new_source, 'strategy': strategy, 'frozen': ['w1', 'b1']}
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
            'validation_sha256': validation_set['sha256'], 'teacher_sha256': teacher_sha,
            'teacher_source': teacher_source,
            'code_edited': new_source != source, 'training_losses': losses,
            'acceptance_policy': action['acceptance_policy'],
            'qwen_call_id': response.call_id, 'reason': action['reason']}
    (candidate / 'training.json').write_text(json.dumps(note, ensure_ascii=False, indent=2), encoding='utf-8')
    hypothesis = action['reason']
    submitted = rnd.submit_candidate(candidate, hypothesis=hypothesis,
        change_summary=f'{strategy}; Adam {action["epochs"]} epochs, steps {before_steps}->{expected}; schedule edit={new_source != source}',
        configuration={'strategy': strategy, 'learning_rate': action['learning_rate'],
                       'epochs': action['epochs'], 'source_sha256': note['source_sha256'],
                       'acceptance_policy': action['acceptance_policy'],
                       'parent_sha256': rnd.parent.checkpoint_sha256},
        seed=student.meta['seed'], checkpoint_paths=['checkpoint.npz', B_TEACHER_PATH],
        code_paths=['train_policy.py', 'training.json'],
        predict_argv=['python3', '/app/harness.py', 'predict'])
    accepted, gate = _accept(parent_validation, submitted, task)
    if accepted and valid_group:
        accepted, gate = _bearing_gate(_metric(parent_validation, task), _metric(submitted, task),
                                       original_path, student, validation_set)
    selected = submitted['artifact_id'] if accepted else rnd.parent.artifact_id
    rnd.finish(selected_artifact_id=selected, hypothesis=hypothesis,
        change_summary=f'{strategy}; candidate measured by host; {gate}',
        reason=f'{action["reason"]}; Qwen specified {action["acceptance_policy"]}; {gate}; model call {response.call_id}',
        configuration={'strategy': strategy, 'candidate_sha256': note['checkpoint_sha256'],
                       'parent_sha256': rnd.parent.checkpoint_sha256, 'accepted': accepted,
                       'learning_rate': action['learning_rate'], 'epochs': action['epochs'],
                       'source_sha256': note['source_sha256'], 'acceptance_policy': action['acceptance_policy'],
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
    winner = int(prediction.argmax())
    tied = np.flatnonzero(prediction == prediction[winner])
    if len(tied) > 1:
        # Match the model's deterministic argmax while satisfying the SDK's unique top-score rule.
        prediction[tied[1:]] = np.nextafter(prediction[winner], 0.0)
    return [{'label': label, 'score': float(score)} for label, score in zip(model.meta['classes'], prediction)]
