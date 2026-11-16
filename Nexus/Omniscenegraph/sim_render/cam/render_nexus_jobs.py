"""Run reviewed render argument lists in one interpreter (reuse imports)."""
import argparse
import gc
import json
from pathlib import Path
import sys

from render_nexus import main as render_main


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jobs',type=Path,required=True)
    a=p.parse_args();jobs=json.loads(a.jobs.read_text())
    for job in jobs:
        print('NEXUS_RENDER_JOB '+job['name'],flush=True)
        sys.argv=['render_nexus.py',*job['arguments']]
        render_main()
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    print('NEXUS_RENDER_JOBS_OK '+str(len(jobs)),flush=True)


if __name__=='__main__':main()
