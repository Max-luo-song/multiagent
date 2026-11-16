"""Run the unchanged independent lane-response criteria on the v2 inputs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/pipeline'),str(ROOT/'third_party/nuplan-devkit')]
import audit_nexus_lane_response as audit
if __name__=='__main__':
    audit.OUT=ROOT/'artifacts/nexus_lane_response_v2_20261027'
    audit.main()
