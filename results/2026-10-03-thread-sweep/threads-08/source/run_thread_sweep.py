"""Run the existing CPU/GPU roundtrip benchmark across CPU thread counts, serially."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('output must not exist')
    root=Path(__file__).resolve().parent
    output=args.output.resolve()
    output.mkdir(parents=True)
    order=[1,4,8,16]
    random.Random(20261003).shuffle(order)
    manifest=dict(started_utc=datetime.now(timezone.utc).isoformat(),thread_order=order,
                  batches=[1,2,4,8,16,32,64,128],trials=5,warmup=20,samples=50,seed=3010,
                  protocol='Existing profile_workloads.py; CPU FP32 input/output including blocking CUDA copies/casts. Four thread counts in one seeded shuffled order; not fully counterbalanced across time. Five seeds per count; not a trained-model quality or HTTP test. GPU experiments must run serially, with no competing inference.')
    def save():
        (output/'sweep.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode())
    save()
    for threads in order:
        run=output/f'threads-{threads:02d}'
        command=[sys.executable,str(root/'profile_workloads.py'),'--output',str(run),
                 '--profiles','roundtrip','--threads',str(threads),'--trials','5','--warmup','20',
                 '--samples','50','--seed','3010','--batch-sizes',*map(str,manifest['batches'])]
        print(f'Start threads={threads}',flush=True)
        subprocess.run(command,cwd=root,check=True)
        meta=json.loads((run/'metadata.json').read_text(encoding='utf-8'))
        archive=run/'source'
        archive.mkdir()
        for name,digest in meta['source_sha256'].items():
            data=(root/name).read_bytes()
            if hashlib.sha256(data).hexdigest()!=digest:
                raise ValueError(f'Source changed during measurement: {name}')
            (archive/name).write_bytes(data)
        subprocess.run([sys.executable,str(root/'select_profile.py'),'--run',str(run),
                        '--output',str(run/'policy-error003.json'),'--min-trials','5','--max-relative-error','0.03'],
                       cwd=root,check=True)
    manifest['completed_utc']=datetime.now(timezone.utc).isoformat()
    save()


if __name__=='__main__':
    main()
