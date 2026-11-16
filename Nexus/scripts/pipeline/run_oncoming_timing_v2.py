"""Same frozen sampler, separate output tree for the timing revision."""
from pathlib import Path
import run_four_hazards as runner
runner.OUT=Path(__file__).resolve().parents[2]/'artifacts/nexus_four_hazards_20261110/oncoming_timing_v2'
if __name__=='__main__':runner.main()
