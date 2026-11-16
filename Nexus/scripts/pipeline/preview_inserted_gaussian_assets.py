"""Inspect the supplied Gaussian vehicles in native CUDA rasterization."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
def main():
 import numpy as np,torch
 from plyfile import PlyData
 from gsplat.rendering import rasterization
 from gsplat.cuda._wrapper import spherical_harmonics
 from PIL import Image,ImageDraw
 out=ROOT/'artifacts/nexus_3dgs/asset_insert_20261028';out.mkdir(exist_ok=True)
 torch.set_num_threads(4)
 contact=Image.new('RGB',(1536,800),'#e7e9ed');draw=ImageDraw.Draw(contact);results=[]
 for row,path in enumerate(sorted(Path('/inspire/hdd3/project/intelligent-driving-agent/public/workspace/hzy/3dgs_pro').glob('*.ply'))):
  v=PlyData.read(path)['vertex'].data;xyz=np.stack([v[k] for k in ('x','y','z')],-1);center=(xyz.min(0)+xyz.max(0))/2;xyz-=center
  means=torch.tensor(xyz,device='cuda');scales=torch.tensor(np.stack([v[f'scale_{k}'] for k in range(3)],-1),device='cuda').exp();quats=torch.tensor(np.stack([v[f'rot_{k}'] for k in range(4)],-1),device='cuda');quats=torch.nn.functional.normalize(quats,dim=-1);alpha=torch.tensor(v['opacity'].copy(),device='cuda').sigmoid()
  dc=torch.tensor(np.stack([v[f'f_dc_{k}'] for k in range(3)],-1),device='cuda');rest=torch.tensor(np.stack([v[f'f_rest_{k}'] for k in range(45)],-1),device='cuda')
  for col,(az,layout) in enumerate([(a,'interleaved') for a in (45,135,225,315)]+[(225,'standard'),(225,'dc')]):
   x,y=np.cos(np.deg2rad(az))*7,np.sin(np.deg2rad(az))*7;position=np.array([x,y,2.5]);forward=-position;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1.]);right/=np.linalg.norm(right);down=np.cross(forward,right);C=np.eye(4);C[:3,:3]=np.c_[right,down,forward];C[:3,3]=position
   direction=means-torch.tensor(position,dtype=torch.float32,device='cuda');direction=direction/direction.norm(dim=-1,keepdim=True)
   if layout=='dc':colors=(dc*.28209479177387814+.5).clamp(0,1)
   else:
    rr=rest.reshape(-1,15,3) if layout=='interleaved' else rest.reshape(-1,3,15).transpose(1,2)
    colors=(spherical_harmonics(3,direction,torch.cat([dc[:,None],rr],1))+.5).clamp(0,1)
   rgb,a,_=rasterization(means=means,quats=quats,scales=scales,opacities=alpha,colors=colors,viewmats=torch.tensor(np.linalg.inv(C),dtype=torch.float32,device='cuda')[None],Ks=torch.tensor([[[460.,0,256.],[0,460.,170.],[0,0,1.]]],device='cuda'),width=512,height=360,packed=False)
   image=(rgb[0]+(1-a[0])*.88).clamp(0,1).cpu().numpy();im=Image.fromarray((image*255).astype('uint8'));label=f'{path.name} az={az} {layout}'
   k=col%3;level=col//3;tile=Image.new('RGB',(512,400),'white');tile.paste(im,(0,30));ImageDraw.Draw(tile).text((10,8),label,fill='black');tile.save(out/f'asset_{path.stem}_{az}_{layout}.jpg');contact.paste(tile,(k*512,(row*2+level)*400)) if False else None
   results.append({'path':str(path),'angle':az,'layout':layout,'image':str(out/f'asset_{path.stem}_{az}_{layout}.jpg')})
 # Four front/back/side views per supplied vehicle.
 board=Image.new('RGB',(2048,800),'white')
 for row,stem in enumerate(['1','3']):
  for col,az in enumerate((45,135,225,315)):
   im=Image.open(out/f'asset_{stem}_{az}_interleaved.jpg');board.paste(im,(col*512,row*400))
 board.save(out/'asset_contact.jpg');(out/'asset_preview.json').write_text(json.dumps(results,indent=2)+'\n');print('ASSET_PREVIEW_COMPLETE',flush=True)
if __name__=='__main__':main()
