"""Fine-tune a PyTRiO 27B LoRA on accepted CPU training decisions."""
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_pytrio_policy import datum
from selftrain.config import LogConfig, SystemConfig, apply_env_overrides
from selftrain.cpu import LLMPolicy
from selftrain.experiment import ExperimentLogger
from selftrain.harness import PyTrioSamplingClient


def main():
    import pytrio as trio

    model = 'Qwen/Qwen3.8-27B'
    config = apply_env_overrides(SystemConfig.load(str(ROOT/'configs/pytrio.yaml'))).harness
    trio.configure(api_key=config.api_key, base_url='https://pytrio.com',
                   request_timeout_seconds=90, retry_max_attempts=1)
    service = trio.ServiceClient()
    caps = next(m for m in service.get_server_capabilities().supported_models if m.model_name == model)
    if not caps.training.available:
        raise RuntimeError('27B training unavailable')
    examples = {}
    for run in ('cpu-qwen-swanlab-full-20261111', 'cpu-qwen-swanlab-full-20261112-rerun1'):
        for file in sorted((ROOT/'out'/run/'swanlab').glob('cpu-*/decisions.jsonl')):
            for line in file.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                if row['decision'] != 'accept' or row['decided_by'] != 'llm':
                    continue
                state = row['state']
                key = json.dumps(state, sort_keys=True, ensure_ascii=False)
                answer = {k:row[k] for k in ('learning_rate','epochs','reason')}
                examples.setdefault(key, dict(state=state, messages=[
                    dict(role='system', content=LLMPolicy.SYSTEM),
                    dict(role='user', content=json.dumps(state, ensure_ascii=False))],
                    answer=json.dumps(answer, ensure_ascii=False)))
    rows = list(examples.values())
    random.Random(42).shuffle(rows)
    if len(rows) < 4:
        raise ValueError('Too few distinct accepted decisions')
    holdout, train = rows[:2], rows[2:18]
    logger = ExperimentLogger(LogConfig(backend='swanlab',workspace='hequan',project='zhixunyinqing',
                              out_dir=str(ROOT/'out/pytrio-cpu-27b'),swanlab_mode='online'),
                              run_name='zhixun-27b-cpu-policy')
    try:
        logger.start(dict(model=model,rank=8,lr=5e-5,train=len(train),holdout=len(holdout)),
                     dict(source='distinct accepted public CPU decisions',seed=42))
        print('Training',len(train),'examples;',len(holdout),'held out.',flush=True)
        trainer = service.create_lora_training_client(base_model=model,rank=8,seed=42)
        tokenizer = trainer.get_tokenizer()
        data = [datum(row,tokenizer,trio) for row in train]
        tokens = sum(len(d.model_input) for d in data)
        if tokens > 12000:
            raise ValueError('Training token cap exceeded')
        steps = 0
        for start in range(0,len(data),2):
            batch = data[start:start+2]
            result = trainer.forward_backward(batch,'cross_entropy').result()
            loss=float(result.metrics['loss_mean'])
            if not math.isfinite(loss):
                raise ValueError('Non-finite training loss')
            trainer.optim_step(trio.AdamParams(learning_rate=5e-5)).result()
            steps += 1
            logger.log(steps,dict(loss=loss,tokens=sum(len(d.model_input) for d in batch)))
            print('Completed step',steps,'loss',round(loss,5),flush=True)
        name='zhixun-cpu-policy-27b-v1'
        path = trainer.save_weights_for_sampler(name=name).result().path
        record=dict(base_model=model,model_path=path,steps=steps,train_tokens=tokens,cloud_url=logger.cloud_url)
        receipt=logger.out_dir/'result.json'
        receipt.write_text(json.dumps(record,indent=2),encoding='utf-8')
        print('Saved',path,flush=True)
        config.pytrio_base_model=model
        config.max_tokens=128
        policy=LLMPolicy(config,client=PyTrioSamplingClient(config,path))
        checks=[]
        for row in holdout:
            lr,ep,_=policy.choose(row['state'])
            expected=json.loads(row['answer'])
            checks.append(dict(source=policy.last_source,learning_rate=lr,epochs=ep,
                               matches=policy.last_source=='llm' and (lr,ep)==(expected['learning_rate'],expected['epochs'])))
        record.update(holdout_checks=checks)
        receipt.write_text(json.dumps(record,indent=2),encoding='utf-8')
        logger.finish(record)
        print('Receipt',receipt,'valid',sum(c['source']=='llm' for c in checks),'of',len(checks),flush=True)
    except Exception as exc:
        logger.abort(type(exc).__name__)
        raise

if __name__ == '__main__':
    main()
