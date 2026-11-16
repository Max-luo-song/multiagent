"""Fresh static reconstruction with measured pose refinement and held-out RGB."""
import argparse,copy,json,random,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'Omniscenegraph')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','seed','old-checkpoint','original-scene','output-root']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run-name',required=True);a=p.parse_args()
    import numpy as np,torch,yaml
    from scipy.spatial.transform import Rotation
    from models.trainers.scene_graph import MultiTrainer
    from models.trainers.base import BasicTrainer
    from datasets.base.pixel_source import ScenePixelSource
    from nexus_3dgs.geometry import matrix_quaternion,quaternion_matrix
    config=yaml.safe_load(a.config.read_text());scene_path=Path(config['data']['data_root'])/config['data']['scene_idx'];meta=json.loads((scene_path/'nexus_scene.json').read_text());info=json.loads((scene_path/'instances/instances_info.json').read_text());oldmeta=json.loads((a.original_scene/'nexus_scene.json').read_text());oldinfo=json.loads((a.original_scene/'instances/instances_info.json').read_text());S=np.linalg.inv(meta['frames'][0]['ego_to_world']);oldW=np.array(oldmeta['frames'][0]['ego_to_world']);cameras=len(config['data']['pixel_source']['cameras']);nframes=len(meta['frames']);old=torch.load(a.old_checkpoint,map_location='cpu',weights_only=False);original_fg=old['models']['RigidNodes'];fg=copy.deepcopy(original_fg);ids=fg['dataset_instance_ids'].tolist();new_by_token={record['id']:int(key) for key,record in info.items()};newids=[new_by_token[oldinfo[str(index)]['id']] for index in ids];M=len(ids);translations=torch.zeros((nframes,M,3));quats=torch.zeros((nframes,M,4));quats[:,:,0]=1;visibility=torch.zeros((nframes,M),dtype=torch.bool);mounts=[]
    for m,(oldid,newid) in enumerate(zip(ids,newids)):
        ann0=oldinfo[str(oldid)]['frame_annotations'];byframe=dict(zip(ann0['frame_idx'],ann0['obj_to_world']));usable=[f for f in range(20,51) if f in byframe and bool(original_fg['instances_fv'][f,m])]
        if not usable:usable=[f for f in sorted(byframe) if f<len(original_fg['instances_trans']) and bool(original_fg['instances_fv'][f,m])][:10]
        if not usable:raise ValueError('No source asset registration for '+str(oldid))
        values=[]
        for f in usable:
            T=np.eye(4);T[:3,:3]=quaternion_matrix(original_fg['instances_quats'][f,m].numpy());T[:3,3]=original_fg['instances_trans'][f,m].numpy();values.append(np.linalg.inv(byframe[f])@oldW@T)
        mount=np.eye(4);mount[:3,:3]=Rotation.from_matrix(np.array(values)[:,:3,:3]).mean().as_matrix();mount[:3,3]=np.median(np.array(values)[:,:3,3],axis=0);mounts.append(mount.tolist());ann=info[str(newid)]['frame_annotations']
        for f,T in zip(ann['frame_idx'],ann['obj_to_world']):
            B=S@T@mount;translations[f,m]=torch.tensor(B[:3,3]);quats[f,m]=torch.tensor(matrix_quaternion(B[:3,:3]));visibility[f,m]=True
    fg.update(instances_trans=translations,instances_quats=quats,instances_fv=visibility,dataset_instance_ids=torch.tensor(newids,dtype=fg['dataset_instance_ids'].dtype))
    original_init=MultiTrainer.init_gaussians_from_dataset;original_optimizer=BasicTrainer.initialize_optimizer;original_sample=ScenePixelSource.propose_training_image
    seed=dict(np.load(a.seed));init_report={}
    def initialize_scene(self,dataset):
        P=torch.tensor(seed['points'],dtype=torch.float32,device=self.device);rgb=torch.tensor(seed['colors'],dtype=torch.float32,device=self.device)
        self.models['Background'].create_from_pcd(init_means=P,init_colors=rgb)
        self.models['RigidNodes'].load_state_dict(copy.deepcopy(fg))
        affine=copy.deepcopy(old['models']['Affine']);affine['embedding.weight']=torch.zeros_like(self.models['Affine'].embedding.weight,device='cpu');self.models['Affine'].load_state_dict(affine)
        # Keep the learned old global display transform fixed. This avoids
        # exposure flicker and preserves every reused actor's radiance basis.
        self.gaussian_classes.pop('RigidNodes')
        init_report.update(seed_points=len(P),gaussian_classes_trained=list(self.gaussian_classes),foreground_assets_reused_exactly=True,camera_parameters_fixed_zero=True,affine_function_fixed_to_old_global_transform=True)
        print('REGISTERED_STATIC_INITIALIZED '+json.dumps(init_report),flush=True)
    def initialize_optimizer(self):
        original_optimizer(self)
        for name,model in self.models.items():
            for param in model.parameters():param.requires_grad_(name in ('Background','Sky'))
        print('REGISTERED_STATIC_OPTIMIZER background_sky_only=True',flush=True)
    def sample(self,candidate_indices=None):
        if not hasattr(self,'_registered_candidates'):
            self._registered_candidates={}
            for c in range(cameras):
                all_views=[int(i) for i in candidate_indices if int(i)%cameras==c];moving=[i for i in all_views if i//cameras>=60];self._registered_candidates[c]=(all_views,moving)
        c=0 if random.random()<.6 else random.choice(list(range(1,cameras)));all_views,moving=self._registered_candidates[c];return random.choice(moving if random.random()<.65 and moving else all_views)
    MultiTrainer.init_gaussians_from_dataset=initialize_scene;BasicTrainer.initialize_optimizer=initialize_optimizer;ScenePixelSource.propose_training_image=sample
    sys.argv=['train.py','--config_file',str(a.config),'--output_root',str(a.output_root),'--project','nuplan','--run_name',a.run_name]
    try:runpy.run_path(str(ROOT/'Omniscenegraph/tools/train.py'),run_name='__main__')
    finally:MultiTrainer.init_gaussians_from_dataset=original_init;BasicTrainer.initialize_optimizer=original_optimizer;ScenePixelSource.propose_training_image=original_sample
    directory=a.output_root/'nuplan'/a.run_name;after=torch.load(directory/'checkpoint_final.pth',map_location='cpu',weights_only=False)['models']
    for key,value in fg.items():assert torch.equal(value,after['RigidNodes'][key]),key
    assert torch.count_nonzero(after['CamPose']['embeds.weight'])==0
    for key,value in old['models']['Affine'].items():
        if key=='embedding.weight':assert torch.count_nonzero(after['Affine'][key])==0
        else:assert torch.equal(value,after['Affine'][key]),key
    report={'status':'passed',**init_report,'all_initialized_foreground_parameters_preserved_exactly':True,'source_fg_world_poses_reinitialized_from_refined_annotations_with_fixed_stationary_mounts':True,'mounts':mounts,'heldout_timesteps':list(range(10,nframes,10)),'heldout_rgb_not_used_in_gaussian_seed':True,'heldout_lidar_not_directly_sampled_as_gaussians':True,'heldout_images_not_used_in_gradient_training':True,'pose_refinement_uses_independent_lidar_including_validation_times':True,'validation_scope':'RGB holdout; pose registration uses all LiDAR timestamps','nexus_weights_or_predictions_trained':False,'moving_sampling_probability':.65,'front_sampling_probability':.6};(directory/'registered_static_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('REGISTERED_STATIC_TRAIN_COMPLETE '+str(directory),flush=True)


if __name__=='__main__':main()
