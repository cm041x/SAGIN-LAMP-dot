"""Train the original HCSAC architectures in the dynamic fleet environment."""
import argparse
import json
from pathlib import Path
import traceback
import torch
import torch_geometric
from Our_experiment.HCSAC.training.dynamic_hcsac_trainer import ROOT,TrainingMode,select_device,train_run,write_csv
from Our_experiment.HCSAC.training.training_split import build_split,episode_manifest

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--modes',default='scratch,finetune');p.add_argument('--train-seeds',default='3001,3002,3003')
    p.add_argument('--episodes',type=int,default=1000);p.add_argument('--max-steps',type=int,default=500)
    p.add_argument('--warmup-steps',type=int,default=2000);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--validation-every',type=int,default=50);p.add_argument('--checkpoint-every',type=int,default=50)
    p.add_argument('--output-dir',default='Our_experiment/HCSAC/data/dynamic_hcsac_baselines/training')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if min(a.episodes,a.max_steps,a.batch_size,a.validation_every,a.checkpoint_every)<1 or a.warmup_steps<0:p.error('Invalid training parameter')
    output=Path(a.output_dir);output=output if output.is_absolute() else ROOT/output;output.mkdir(parents=True,exist_ok=True)
    (output/'failed_training_runs.jsonl').touch(exist_ok=True)
    # Freeze every paired seed schedule before the first training run starts.
    split=build_split()
    (output/'training_split.json').write_text(json.dumps(split,indent=2)+'\n')
    (output/'validation_manifest.json').write_text(json.dumps(split['validation_combinations'],indent=2)+'\n')
    seeds=list(map(int,a.train_seeds.split(',')))
    for seed in seeds:write_csv(output/f'episode_manifest_seed_{seed}.csv',episode_manifest(seed,a.episodes,split))
    for seed in seeds:
        for mode in map(TrainingMode,a.modes.split(',')):
            try:
                device=select_device(a.device)
                path=output/mode.value/f'seed_{seed}'/'run_manifest.json'
                old=json.loads(path.read_text()) if path.exists() else {}
                previous=old.get('total_elapsed_seconds',old.get('elapsed_seconds_this_invocation',0))
                if device=='cuda':torch.cuda.reset_peak_memory_stats()
                result=train_run(mode,seed,a.episodes,a.max_steps,a.warmup_steps,a.batch_size,a.validation_every,a.checkpoint_every,output,device,a.resume)
                result.update(total_elapsed_seconds=previous+result['elapsed_seconds_this_invocation'],
                              torch_geometric_version=torch_geometric.__version__,
                              gpu_name=torch.cuda.get_device_name() if device=='cuda' else None,
                              peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
                path.write_text(json.dumps(result,indent=2)+'\n')
            except Exception:
                with (output/'failed_training_runs.jsonl').open('a') as f:f.write(json.dumps({'mode':mode.value,'seed':seed,'traceback':traceback.format_exc()})+'\n')
                raise
if __name__=='__main__':main()
