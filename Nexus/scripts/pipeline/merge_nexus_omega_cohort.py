"""Assemble all predeclared seeds from identical-config runs without resampling."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validation-request-only',action='store_true')
    a=p.parse_args()
    first=a.runs[0]
    request=json.loads((first/'request.json').read_text())
    if a.validation_request_only:
        request.update(seeds=[1,2],render_seed=1)
        if a.output.exists():raise FileExistsError(a.output)
        a.output.write_text(json.dumps(request,indent=2)+'\n')
        print('VALIDATION_REQUEST_READY',flush=True);return
    a.output.mkdir(parents=True,exist_ok=False)
    reference=json.loads((first/'protocol.json').read_text())
    expected_request=copy.deepcopy(request)
    for key in ('seeds','render_seed'):expected_request.pop(key,None)
    combined={'status':'complete','records':[],'pairs':[],'model_parameters_unchanged':True,
              'assembled_without_new_inference':True,'source_runs':[]}
    keys=set();seeds=set()
    for root in a.runs:
        run=json.loads((root/'run.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
        assert run['status']=='complete' and run['model_parameters_unchanged']
        for key in ('config','history_velocity','native_constraints','profile','checkpoint_sha256','raw_features_sha256','scenario','background_replay','drivable_sdf'):
            assert protocol[key]==reference[key],key
        other_request=json.loads((root/'request.json').read_text())
        for key in ('seeds','render_seed'):other_request.pop(key,None)
        assert other_request==expected_request
        for key,checksum in reference['source_sha256'].items():
            if key.endswith('run_nexus_omega_v2.py'):continue  # metadata-only driver updates are separately archived
            assert protocol['source_sha256'][key]==checksum,key
        for row in run['records']:
            key=(row['method'],row['seed'],row['variant'])
            if key in keys:raise ValueError('Repeated seed/variant: '+str(key))
            keys.add(key);seeds.add(row['seed'])
            source=root/row['path'];target=a.output/row['path']
            shutil.copytree(source,target)
            for path in source.rglob('*'):
                if path.is_file():assert sha(path)==sha(target/path.relative_to(source))
            item=copy.deepcopy(row);item['source_run']=str(root.resolve());combined['records'].append(item)
        combined['pairs']+=run['pairs']
        combined['source_runs'].append({'path':str(root.resolve()),'run_sha256':sha(root/'run.json'),
                                        'protocol_sha256':sha(root/'protocol.json')})
    assert seeds=={0,1,2}
    assert len(keys)==6 and all(('anchor',s,v) in keys for s in seeds for v in ('normal','event'))
    combined['records'].sort(key=lambda r:(r['seed'],r['variant']))
    reference['seeds']=[0,1,2];reference['cohort_sources']=combined['source_runs'];reference['planned_render_seed']=0
    request.update(seeds=[0,1,2],render_seed=0)
    for name in ('normal_condition.json','event_condition.json'):
        shutil.copy2(first/name,a.output/name)
    shutil.copytree(first/'sources',a.output/'sources')
    for name,value in (('run.json',combined),('protocol.json',reference),('request.json',request)):
        (a.output/name).write_text(json.dumps(value,indent=2)+'\n')
    print('COHORT_COMPLETE',sorted(seeds),flush=True)


if __name__=='__main__':main()
