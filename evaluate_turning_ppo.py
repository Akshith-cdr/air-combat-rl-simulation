"""Paired held-out baseline/candidate evaluation. This file never trains."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import numpy as np
import torch

from turning_experiment import (BASE_MODEL,BASE_STATS,ExperimentConfig,load_pair,read_bundle,
                                sample_scenario,sha256,verify_baseline)


def held_out_manifest(pairs=5,seed=91001):
    rng=np.random.default_rng(seed)
    result=[]
    families=('ahead','left','right','behind_left','behind_right')
    for index in range(pairs):
        scenario=sample_scenario(rng,(30,90,150,210,270,330),families[index%5])
        result.extend([asdict(scenario),asdict(scenario.mirror())])
    return result


def _episode_impl(model,norm,scenario,seed,policy_seed,deterministic,max_steps,observer=None):
    raw=norm.venv.envs[0].unwrapped
    observation,_=raw.reset(seed=seed,options={'turning_scenario':scenario})
    if observer is not None:
        from turning_telemetry import initial_state
        saved_initial_state=initial_state(raw,observation,seed,policy_seed)
    torch.manual_seed(policy_seed)  # After loading, identically for each policy.
    initial=np.array(raw.sim.player.position,dtype=float)
    previous=initial.copy()
    previous_heading=raw.sim.player.yaw
    path=area=absolute_turn=net_turn=abs_bank=0.0
    banked=opposed=left_banked=low_speed=low_altitude=0
    run=longest_run=0
    previous_bank_sign=0
    aligned_run=0
    first_alignment=sustained_alignment=None
    requests=launches=rejections=0
    reward_total=0.0
    trace=[]
    info={}
    for step in range(1,max_steps+1):
        normalized=norm.normalize_obs(observation[None,:])
        if observer is not None:
            input_observation=observation.copy()
        action,_=model.predict(normalized,deterministic=deterministic)
        if observer is not None:
            from turning_telemetry import policy_snapshot
            distribution=policy_snapshot(model)
        observation,reward,terminated,truncated,info=raw.step(action[0])
        p=raw.sim.player
        position=np.array(p.position,dtype=float)
        path+=float(np.linalg.norm(position-previous))
        previous=position
        error=abs((float(observation[15])+180)%360-180)
        area+=error*raw.simulation_dt
        delta=(p.yaw-previous_heading+180)%360-180
        previous_heading=p.yaw
        absolute_turn+=abs(delta)
        net_turn+=delta
        abs_bank+=abs(p.roll)
        banked+=abs(p.roll)>45
        left_banked+=p.roll<0
        opposed+=(abs(p.roll)>15 and p.roll*p.yaw_input>0 and abs(p.yaw_input)>.05)
        sign=int(np.sign(p.roll)) if abs(p.roll)>45 else 0
        run=run+1 if sign!=0 and sign==previous_bank_sign else (1 if sign else 0)
        longest_run=max(longest_run,run)
        previous_bank_sign=sign
        low_speed+=p.airspeed<150
        low_altitude+=p.altitude<250
        if error<=10:
            aligned_run+=1
            if first_alignment is None: first_alignment=step*raw.simulation_dt
            if aligned_run>=10 and sustained_alignment is None:
                sustained_alignment=(step-9)*raw.simulation_dt
        else:
            aligned_run=0
        weapon=info['weapon']
        requests+=int(weapon['fire_request'])
        launches+=int(weapon['valid_fire_attempt'])
        rejections+=int(weapon['invalid_fire_attempt'])
        reward_total+=reward
        trace.append({'step':step,'bearing_error_deg':float(observation[15]),'bank_deg':p.roll,
                      'heading_left_deg':p.yaw,'action':action[0].tolist(),
                      'applied_controls':[p.pitch_input,p.roll_input,p.yaw_input,p.speed_input],
                      'position':position.tolist()})
        if observer is not None:
            trace[-1]['firing_telemetry']={
                'schema_version':1,'phase':'before_physics','deterministic':deterministic,
                'raw_observation':input_observation.tolist(),
                'normalized_observation':normalized[0].tolist(),
                'policy_distribution':distribution,'selected_action':action[0].tolist(),
                'clipped_action':observer.clipped,'applied_action':observer.applied,
                'gate':observer.record,
            }
        if terminated or truncated:
            break
    terminal=Counter(event.reason for event in raw.sim.missile_events)
    displacement=float(np.linalg.norm(previous-initial))
    result={'steps':step,'duration_s':step*raw.simulation_dt,'first_alignment_s':first_alignment,
            'sustained_alignment_s':sustained_alignment,'accumulated_absolute_bearing_deg_s':area,
            'mean_absolute_bearing_deg':area/(step*raw.simulation_dt),
            'final_absolute_bearing_deg':error,'path_length_m':path,'displacement_m':displacement,
            'excess_over_displacement_m':max(0,path-displacement),
            'path_displacement_ratio':path/max(displacement,1e-9),
            'absolute_heading_rotation_deg':absolute_turn,'net_heading_rotation_deg':net_turn,
            'absolute_turn_revolutions_proxy':absolute_turn/360,
            'mean_absolute_bank_deg':abs_bank/step,'large_bank_fraction':banked/step,
            'left_bank_fraction':left_banked/step,'opposing_bank_yaw_fraction':opposed/step,
            'longest_same_side_large_bank_s':longest_run*raw.simulation_dt,
            'low_speed_steps':low_speed,'low_altitude_steps':low_altitude,
            'fire_requests':requests,'authorized_launches':launches,'rejected_requests':rejections,
            'missile_terminal_counts':dict(terminal),'missile_hits':terminal['hit'],
            'missile_expiries':terminal['expired'],
            'completed_nonhit_missiles':sum(v for k,v in terminal.items() if k!='hit'),
            'pending_missiles':len(raw.sim.active_missiles),
            'episode_success':bool(raw.sim.missile_hit),
            'termination':info.get('terminal_reason') or 'evaluation_step_limit',
            'safety_termination':bool(raw.sim.game_over and not raw.sim.missile_hit),
            'base_reward_total':reward_total}
    if observer is not None:
        from turning_telemetry import summarize_firing
        result['initial_state']=saved_initial_state
        result['firing']=summarize_firing([row['firing_telemetry'] for row in trace])
        result['firing'].update(hits=terminal['hit'],misses=result['completed_nonhit_missiles'],
                               expiries=terminal['expired'],terminal_reasons=dict(terminal),
                               pending_missiles=len(raw.sim.active_missiles))
    return result,trace


def episode(model,norm,scenario,seed,policy_seed,deterministic,max_steps,instrument_firing=False):
    if not instrument_firing:
        return _episode_impl(model,norm,scenario,seed,policy_seed,deterministic,max_steps)
    from turning_telemetry import GateObserver
    with GateObserver(norm.venv.envs[0].unwrapped) as observer:
        return _episode_impl(model,norm,scenario,seed,policy_seed,deterministic,max_steps,observer)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True,help='Completed candidate bundle directory')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pairs',type=int,default=5)
    parser.add_argument('--max-steps',type=int,default=2000)
    parser.add_argument('--scenario-seed',type=int,default=91001)
    parser.add_argument('--policy-seed',type=int,default=73001)
    parser.add_argument('--mode',choices=['deterministic','sampled','both'],default='both')
    parser.add_argument('--instrument-firing',action='store_true',
                        help='Opt-in firing telemetry and schema v2 reporting; no action changes.')
    parser.add_argument('--common-horizon-steps',type=int,default=100,
                        help='Instrumented bearing comparison prefix, capped to available steps.')
    args=parser.parse_args()
    from training_rules import apply_training_launch_rules
    apply_training_launch_rules()
    if args.pairs<1 or args.max_steps<1 or args.common_horizon_steps<1:
        parser.error('pairs and max-steps must be positive')
    metadata=read_bundle(args.candidate)
    verify_baseline()
    if metadata['baseline_hashes']!=verify_baseline():
        raise ValueError('Candidate baseline provenance differs')
    source=ExperimentConfig(**metadata['config'])
    # Both policies use the same broad initial states, original reward and
    # identical guided weapon settings. Training shaping is not an eval score.
    config=replace(source,variant='A',bearing_progress_weight=0,update_normalization=False,
                   max_episode_steps=args.max_steps,device='cpu')
    cases=held_out_manifest(args.pairs,args.scenario_seed)
    args.output.mkdir(parents=True,exist_ok=False)
    before={str(p):sha256(p) for p in (BASE_MODEL,BASE_STATS,args.candidate/'candidate.zip',
                                     args.candidate/'candidate_vecnormalize.pkl')}
    protocol={'candidate_bundle':str(args.candidate),
        'candidate_metadata':metadata,'evaluation_config':asdict(config),'scenarios':cases,
        'scenario_seed':args.scenario_seed,'policy_seed':args.policy_seed,'mode':args.mode,
        'note':'Held-out heading grid; mirrored initial geometry, legacy enemy conventions unchanged. '
               'Path excess, rotation and bank/yaw opposition are proxies, not proof of tactical waste.'}
    if args.instrument_firing:
        config_text=json.dumps(asdict(config),sort_keys=True,indent=2)+'\n'
        with (args.output/'effective_config.json').open('x') as stream:
            stream.write(config_text)
        protocol.update(schema_version=2,instrument_firing=True,
            common_horizon_steps=args.common_horizon_steps,
            effective_config_sha256=sha256(args.output/'effective_config.json'),
            candidate_bundle_sha256=sha256(args.candidate/'bundle.json'),
            source_hashes={name:sha256(name) for name in
                ('evaluate_turning_ppo.py','turning_telemetry.py','turning_experiment.py',
                 'fighter_jet_env.py','hybrid_ppo.py','updated_chase_jet_sim_v3.py','projectile_motion.py')})
        with (args.output/'protocol.json').open('x') as stream:
            stream.write(json.dumps(protocol,indent=2)+'\n')
    else:
        (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[]
    comparison_traces={}
    output_mode='x' if args.instrument_firing else 'w'
    with (args.output/'episodes.jsonl').open(output_mode) as episodes,(args.output/'steps.jsonl').open(output_mode) as traces:
        for name,model_path,stats_path in [('baseline',BASE_MODEL,BASE_STATS),
                ('candidate',args.candidate/'candidate.zip',args.candidate/'candidate_vecnormalize.pkl')]:
            model,norm=load_pair(model_path,stats_path,config)
            try:
                modes=['deterministic','sampled'] if args.mode=='both' else [args.mode]
                for mode in modes:
                    for index,scenario in enumerate(cases):
                        extra={'instrument_firing':True} if args.instrument_firing else {}
                        result,trace=episode(model,norm,scenario,args.scenario_seed+index,
                                             args.policy_seed+index,mode=='deterministic',args.max_steps,**extra)
                        if args.instrument_firing:
                            comparison_traces[name,mode,index]=[
                                {'bearing_error_deg':sample['bearing_error_deg']} for sample in trace]
                        row={'policy':name,'mode':mode,'case':index,**result}
                        episodes.write(json.dumps(row)+'\n')
                        episodes.flush()
                        for sample in trace:
                            traces.write(json.dumps({'policy':name,'mode':mode,'case':index,**sample})+'\n')
                        rows.append(row)
            finally:
                norm.close()
    assert before=={p:sha256(p) for p in before}
    summary={}
    for name in ('baseline','candidate'):
        for mode in ('deterministic','sampled'):
            group=[r for r in rows if r['policy']==name and r['mode']==mode]
            if not group: continue
            keys=['mean_absolute_bearing_deg','accumulated_absolute_bearing_deg_s','path_length_m',
                  'excess_over_displacement_m','mean_absolute_bank_deg','opposing_bank_yaw_fraction',
                  'longest_same_side_large_bank_s','absolute_turn_revolutions_proxy']
            times=[r['sustained_alignment_s'] for r in group if r['sustained_alignment_s'] is not None]
            summary[name+'/'+mode]={'episodes':len(group),'successes':sum(r['episode_success'] for r in group),
                'safety_terminations':sum(r['safety_termination'] for r in group),
                'sustained_alignment_count':len(times),
                'median_sustained_alignment_s':float(np.median(times)) if times else None,
                'mean':{key:float(np.mean([r[key] for r in group])) for key in keys},
                'counts':{key:sum(r[key] for r in group) for key in
                          ['fire_requests','authorized_launches','rejected_requests','missile_hits',
                           'completed_nonhit_missiles','missile_expiries','pending_missiles']}}
            if args.instrument_firing:
                from turning_telemetry import aggregate_firing
                item=summary[name+'/'+mode]
                item['mean_episode_longest_bank_s']=item['mean'].pop('longest_same_side_large_bank_s')
                item['max_episode_longest_bank_s']=max(r['longest_same_side_large_bank_s'] for r in group)
                item['firing']=aggregate_firing([row['firing'] for row in group])
    result={'groups':summary,'hashes_before_after':before}
    if args.instrument_firing:
        from turning_telemetry import paired_bearing_metrics
        result.update(schema_version=2,bearing_comparisons=paired_bearing_metrics(
            comparison_traces,args.common_horizon_steps,metadata.get('simulation_dt_seconds',.05)))
        with (args.output/'summary.json').open('x') as stream:
            stream.write(json.dumps(result,indent=2)+'\n')
    else:
        (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
