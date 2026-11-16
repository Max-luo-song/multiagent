"""Inference-only SegFormer from an official HF state, using existing PyTorch.

Architecture/config reference: Hugging Face Transformers v4.44.2 SegFormer
(Apache-2.0), NVIDIA segformer-b0-finetuned-ade-512-512. This functional
implementation needs no transformers/mmcv/timm installation.
"""
import json
import math
from pathlib import Path
import struct

import numpy as np
import torch
from torch.nn import functional as F


def read_safetensors(path,device):
    raw=Path(path).read_bytes();size=struct.unpack('<Q',raw[:8])[0]
    header=json.loads(raw[8:8+size]);offset=8+size
    dtypes={'F32':np.float32,'F16':np.float16,'I64':np.int64,'I32':np.int32}
    result={}
    for key,value in header.items():
        if key=='__metadata__':continue
        a,b=value['data_offsets']
        result[key]=torch.from_numpy(np.frombuffer(raw[offset+a:offset+b],dtype=dtypes[value['dtype']]).copy().reshape(value['shape'])).to(device)
    return result


class SegformerInference:
    def __init__(self,root,device='cuda'):
        self.root=Path(root);self.config=json.loads((self.root/'config.json').read_text())
        self.processor=json.loads((self.root/'preprocessor_config.json').read_text())
        self.device=device;self.weights=read_safetensors(self.root/'model.safetensors',device)
        self.used=set()

    def tensor(self,key):
        self.used.add(key);return self.weights[key]

    def linear(self,x,key):
        return F.linear(x,self.tensor(key+'.weight'),self.tensor(key+'.bias'))

    def conv(self,x,key,stride=1,padding=0,groups=1):
        return F.conv2d(x,self.tensor(key+'.weight'),self.tensor(key+'.bias') if key+'.bias' in self.weights else None,
                        stride=stride,padding=padding,groups=groups)

    def norm(self,x,key):
        return F.layer_norm(x,(x.shape[-1],),self.tensor(key+'.weight'),self.tensor(key+'.bias'),self.config['layer_norm_eps'])

    @torch.no_grad()
    def __call__(self,pixels):
        """pixels: uint8 RGB [H,W,3], resized with the official PIL preprocessing."""
        from PIL import Image
        size=int(self.processor['size'])
        image=np.array(Image.fromarray(pixels).resize((size,size),Image.Resampling.BILINEAR))
        x=torch.as_tensor(image,device=self.device,dtype=torch.float32).permute(2,0,1)[None]/255
        mean=torch.tensor(self.processor['image_mean'],device=self.device)[None,:,None,None]
        std=torch.tensor(self.processor['image_std'],device=self.device)[None,:,None,None];x=(x-mean)/std
        features=[]
        for stage in range(4):
            key=f'segformer.encoder.patch_embeddings.{stage}'
            x=self.conv(x,key+'.proj',stride=self.config['strides'][stage],padding=self.config['patch_sizes'][stage]//2)
            B,C,H,W=x.shape;x=self.norm(x.flatten(2).transpose(1,2),key+'.layer_norm')
            heads=self.config['num_attention_heads'][stage];ratio=self.config['sr_ratios'][stage]
            for layer in range(self.config['depths'][stage]):
                key=f'segformer.encoder.block.{stage}.{layer}';y=self.norm(x,key+'.layer_norm_1')
                ak=key+'.attention.self'
                q=self.linear(y,ak+'.query').reshape(B,-1,heads,C//heads).transpose(1,2)
                kv=y
                if ratio>1:
                    kv=self.conv(y.transpose(1,2).reshape(B,C,H,W),ak+'.sr',stride=ratio)
                    kv=self.norm(kv.flatten(2).transpose(1,2),ak+'.layer_norm')
                k=self.linear(kv,ak+'.key').reshape(B,-1,heads,C//heads).transpose(1,2)
                v=self.linear(kv,ak+'.value').reshape(B,-1,heads,C//heads).transpose(1,2)
                attention=(q@k.transpose(-1,-2)/math.sqrt(C//heads)).softmax(-1)
                y=(attention@v).transpose(1,2).reshape(B,-1,C)
                x=x+self.linear(y,key+'.attention.output.dense')
                y=self.linear(self.norm(x,key+'.layer_norm_2'),key+'.mlp.dense1')
                hidden=y.shape[-1]
                y=self.conv(y.transpose(1,2).reshape(B,hidden,H,W),key+'.mlp.dwconv.dwconv',padding=1,groups=hidden)
                y=F.gelu(y.flatten(2).transpose(1,2));x=x+self.linear(y,key+'.mlp.dense2')
            x=self.norm(x,f'segformer.encoder.layer_norm.{stage}').transpose(1,2).reshape(B,C,H,W)
            features.append(x)
        outputs=[]
        for stage,x in enumerate(features):
            B,C,H,W=x.shape
            y=self.linear(x.flatten(2).transpose(1,2),f'decode_head.linear_c.{stage}.proj')
            y=y.transpose(1,2).reshape(B,self.config['decoder_hidden_size'],H,W)
            outputs.append(F.interpolate(y,size=features[0].shape[-2:],mode='bilinear',align_corners=False))
        y=self.conv(torch.cat(outputs[::-1],dim=1),'decode_head.linear_fuse')
        key='decode_head.batch_norm'
        y=F.batch_norm(y,self.tensor(key+'.running_mean'),self.tensor(key+'.running_var'),
                       self.tensor(key+'.weight'),self.tensor(key+'.bias'),training=False,eps=1e-5)
        logits=self.conv(F.relu(y),'decode_head.classifier')
        logits=F.interpolate(logits,size=pixels.shape[:2],mode='bilinear',align_corners=False)
        probs=logits.softmax(1)[0];confidence,labels=probs.max(0)
        missing=set(self.weights)-self.used-{key+'.num_batches_tracked'}
        if missing:raise ValueError('Unused model weights: '+str(sorted(missing)))
        return labels.cpu().numpy().astype(np.uint8),confidence.cpu().numpy(),probs[2].cpu().numpy()
