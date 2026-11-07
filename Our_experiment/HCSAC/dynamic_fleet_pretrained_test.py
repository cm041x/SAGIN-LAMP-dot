"""Run the same diagnostic using the unmodified pretrained CNN and GCN actors."""
import argparse
import traceback
from pathlib import Path
from Our_experiment.HCSAC.dynamic_fleet_smoke_test import ROOT, run

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-steps',type=int,default=200)
    p.add_argument('--output-dir',default='Our_experiment/HCSAC/data/dynamic_fleet_pretrained')
    a=p.parse_args()
    out=Path(a.output_dir)
    if not out.is_absolute(): out=ROOT/out
    out.mkdir(parents=True,exist_ok=True)
    try:
        summary=run(a.seed,a.max_steps,a.output_dir,'pretrained')
        if not summary['pretrained_smoke_passed']:
            raise RuntimeError('Pretrained smoke acceptance failed; see summary')
    except Exception:
        (out/'traceback.txt').write_text(traceback.format_exc())
        raise

if __name__=='__main__': main()
