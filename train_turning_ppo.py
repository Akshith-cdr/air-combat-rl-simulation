"""Manual-only PPO fine-tuning entry point. Never called by validation/tests."""

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import zipfile

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from turning_experiment import (BASE_MODEL,BASE_STATS,load_config,load_pair,read_bundle,
                                reserve_run,sha256,validate_device,verify_baseline)
from turning_checkpoints import save_bundle,validate_resume
from training_rules import apply_training_launch_rules


class ExperimentLogs(BaseCallback):
    def __init__(self,run,config,source_steps,parent):
        super().__init__()
        self.run,self.config,self.source_steps,self.parent=run,config,source_steps,parent
        self.last_save=source_steps
        self.rows=[]
        self.file=(run/'reward_metrics.csv').open('w',newline='')
        self.writer=None

    def flush_metrics(self):
        if not self.rows:
            return
        keys=sorted(self.rows[0])
        values={key:float(np.mean([r[key] for r in self.rows])) for key in keys}
        if self.writer is None:
            self.writer=csv.DictWriter(self.file,fieldnames=['num_timesteps']+keys)
            self.writer.writeheader()
        self.writer.writerow({'num_timesteps':self.model.num_timesteps,**values})
        self.file.flush()
        for key,value in values.items():
            self.logger.record('reward_components/'+key,value)
        self.rows=[]

    def _on_step(self):
        self.rows.extend(info['reward_breakdown'] for info in self.locals['infos'])
        if len(self.rows)>=100:
            self.flush_metrics()
        return True

    def _on_rollout_start(self):
        # Previous rollout has now been optimized; avoid presenting a partial
        # rollout checkpoint as an exact continuation of an update.
        self.flush_metrics()
        if self.model.num_timesteps-self.last_save>=self.config.checkpoint_every:
            save_bundle(self.run/'checkpoints'/f'step_{self.model.num_timesteps:012d}',
                        self.model,self.training_env,self.config,self.source_steps,self.parent)
            self.last_save=self.model.num_timesteps

    def _on_training_end(self):
        self.flush_metrics()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--device')
    parser.add_argument('--additional-steps',type=int)
    parser.add_argument('--resume',type=Path,help='Completed bundle directory; a NEW run-id is required')
    parser.add_argument('--init-model',type=Path,default=BASE_MODEL,
                        help='Stage 4 hybrid model to fine-tune (default: the supplied Stage 4 model)')
    parser.add_argument('--init-vecnormalize',type=Path,
                        help='VecNormalize statistics of --init-model (default: <model>_vecnormalize.pkl)')
    args=parser.parse_args()
    apply_training_launch_rules()
    if args.init_vecnormalize is None:
        args.init_vecnormalize=args.init_model.with_name(args.init_model.stem+'_vecnormalize.pkl')
    config=load_config(args.config)
    changes={}
    if args.device is not None: changes['device']=args.device
    if args.additional_steps is not None: changes['total_timesteps']=args.additional_steps
    config=replace(config,**changes)
    validate_device(config.device)
    hashes=verify_baseline()
    if args.resume:
        metadata=read_bundle(args.resume)
        validate_resume(metadata,config)
        model_path,stats_path=args.resume/'candidate.zip',args.resume/'candidate_vecnormalize.pkl'
    else:
        model_path,stats_path=args.init_model,args.init_vecnormalize
        if (sha256(model_path),sha256(stats_path))!=(sha256(BASE_MODEL),sha256(BASE_STATS)):
            print('NOTE: starting from a Stage 4 model other than the supplied one; the result will '
                  'not be bit-identical to experiments/turning_improvement/A_seed62001_run01.')
    with zipfile.ZipFile(model_path) as archive:
        if 'policy.optimizer.pth' not in archive.namelist():
            raise ValueError('Optimizer state unavailable; this initialization strategy requires it')
    # Load/validate before reserving output. No original files are written.
    model,norm=load_pair(model_path,stats_path,config,training=True,device=config.device)
    callback=None
    try:
        if not model.policy.optimizer.state:
            raise ValueError('Checkpoint optimizer state is empty')
        source_steps=int(model.num_timesteps)
        run=reserve_run(args.run_id)
        model.seed=config.seed
        model.set_random_seed(config.seed)
        (run/'run.json').write_text(json.dumps({'config':asdict(config),'baseline_hashes':hashes,
            'initial_model':str(model_path),'initial_statistics':str(stats_path),
            'source_timesteps':source_steps,'strategy':'optimizer-restored fine-tune, not exact replay'},indent=2)+'\n')
        model.set_logger(configure(str(run/'logs'),['stdout','csv']))
        callback=ExperimentLogs(run,config,source_steps,args.resume)
        save_bundle(run/'initial',model,norm,config,source_steps,args.resume,kind='initial')
        print(f'Additional requested steps: {config.total_timesteps}; source count: {source_steps}; '
              f'rollouts round the actual count upward to multiples of {config.n_steps}.')
        model.learn(total_timesteps=config.total_timesteps,reset_num_timesteps=False,callback=callback)
        metadata=save_bundle(run/'final',model,norm,config,source_steps,args.resume,kind='final')
        print(json.dumps({'run':str(run),'requested_additional_steps':config.total_timesteps,
                          'actual_additional_steps':metadata['actual_additional_timesteps'],
                          'final_timesteps':metadata['num_timesteps']},indent=2))
    except KeyboardInterrupt:
        print('Interrupted. Resume from the latest completed checkpoints/step_* bundle under a NEW run-id. '
              'Partial rollouts and hidden staging bundles are not resumable. No final checkpoint was published.')
        raise
    finally:
        if callback is not None: callback.file.close()
        norm.close()
        verify_baseline()


if __name__=='__main__':
    main()
