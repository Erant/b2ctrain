#!/usr/bin/env python3
"""Transfer edits from a view-shaped UV sheet (or its extra sheet) to its original atlas.

This is a direct UV-to-UV barycentric lookup, with no camera reprojection or
visibility weighting. Unowned surfaces and unused source texels remain exact.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from view_atlas import cast, scene_for, sample


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--layout',type=Path,required=True)
    ap.add_argument('--edited',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--reference-sheet',type=Path,help='Transfer edited-minus-reference deltas, preserving source detail and making unchanged edits exact')
    ap.add_argument('--source-texture',type=Path,help='Texture to edit instead of the layout\'s recorded source (chains the extra sheet onto the first sheet\'s result)')
    a=ap.parse_args()
    meta=json.loads((a.layout/'atlas.json').read_text())
    mesh=o3d.io.read_triangle_mesh(str(Path(meta['source'])/'mesh_uv.obj'),enable_post_processing=False)
    old=np.asarray(mesh.triangle_uvs).reshape(-1,3,2)
    new=np.load(a.layout/'face_uv.npy'); owner=np.load(a.layout/'face_panel.npy')
    if old.shape!=new.shape or len(owner)!=len(old): raise ValueError('Layout/source topology mismatch')
    source=cv2.imread(str(a.source_texture or meta['source_texture'])); edited=cv2.imread(str(a.edited))
    if source is None or edited is None: raise ValueError('Missing texture')
    if edited.shape[:2]!=(meta.get('height',meta.get('res')),meta.get('width',meta.get('res'))): raise ValueError('Edited image must keep original layout dimensions')
    reference=cv2.imread(str(a.reference_sheet)) if a.reference_sheet else None
    if a.reference_sheet and (reference is None or reference.shape!=edited.shape): raise ValueError('Reference sheet must match edited image')
    h,w=source.shape[:2]; result=source.copy(); changed=np.zeros((h,w),np.uint8)
    flat=np.concatenate([old.reshape(-1,2),np.zeros((len(old)*3,1))],1)
    scene=scene_for(flat,np.arange(len(old)*3).reshape(-1,3))
    for y in range(0,h,128):
        yy,xx=np.meshgrid(np.arange(y,min(y+128,h)),np.arange(w),indexing='ij')
        origins=np.stack([(xx+.5)/w,1-(yy+.5)/h,np.ones_like(xx)],-1)
        hit=cast(scene,origins,[0,0,-1]); valid=np.isfinite(hit['t_hit'].numpy()); pid=hit['primitive_ids'].numpy(); pid[~valid]=0
        valid &= owner[pid]>=0
        b=hit['primitive_uvs'].numpy(); bw=np.stack([1-b.sum(-1),b[...,0],b[...,1]],-1)
        uv=(new[pid]*bw[...,None]).sum(-2)
        color=sample(edited,uv)
        sl=slice(y,y+len(yy))
        if reference is not None:
            color=np.clip(source[sl].astype(np.int16)+color.astype(np.int16)-sample(reference,uv).astype(np.int16),0,255).astype(np.uint8)
            valid &= np.any(color!=source[sl],axis=-1)
        result[sl][valid]=color[valid]; changed[sl]=valid*255
    # Refresh only gutters whose closest surface texel received an edit.
    mask=cv2.imread(str(Path(meta['source'])/'mask.png'),cv2.IMREAD_GRAYSCALE)
    dist,labels=cv2.distanceTransformWithLabels(255-mask,cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    lut=np.zeros(labels.max()+1,np.int64); lut[labels[mask>0]]=np.flatnonzero(mask)
    nearest=lut[labels]; gutter=(mask==0)&(dist<=meta['pad'])&(changed.reshape(-1)[nearest]>0)
    result[gutter]=result.reshape(-1,3)[nearest[gutter]]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(a.output),result)
    cv2.imwrite(str(a.output.with_suffix('.mask.png')),changed)
    error=np.abs(result.astype(float)-source.astype(float))
    metrics=dict(edited_texels=int((changed>0).sum()),owned_mean_absolute_error=float(error[changed>0].mean()) if changed.any() else 0.0,unowned_surface_max_error=float(error[(mask>0)&(changed==0)].max(initial=0)))
    a.output.with_suffix('.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(metrics)


if __name__=='__main__': main()
