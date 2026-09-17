#!/usr/bin/env python3
"""Build a view-shaped UV atlas, preserving geometry and the source texture.

Requires numpy, opencv-python and open3d. No model inference is performed.
Front/back panels contain contextual pixels outside their ownership masks; only
owned triangles address them. Hidden/overlapping faces retain source UVs in a
separate fallback panel. Input mesh must be triangulated with one UV per corner.

Beyond the six axis panels, --extra N oblique full-body views and --head M
close-up head views are chosen greedily by the effective texel density
((px/m)^2 x facing cosine) they add; a triangle moves to a view only when it gets
more texels there. Every panel's context is a render of the whole mesh, so an
image model always sees a person. --layout multi writes one sheet per group
(<output>, extra/, head/); --layout single packs every panel onto one wide sheet
so one denoise sees all views at once. Reserve surface no candidate direction
sees at all (the TSDF's inner shell) is reported as interior.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def scene_for(v, f):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(v.astype(np.float32)),
                        o3d.core.Tensor(f.astype(np.uint32)))
    return scene


def cast(scene, origins, direction):
    rays = np.concatenate([origins, np.broadcast_to(direction, origins.shape)], -1)
    return scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))


def overlap_pairs(tri):
    """Strict positive-area triangle intersections; shared edges are allowed.

Spatial bins enumerate all bounding-box candidates, then a separating-axis test
checks actual triangles. This checks continuous UVs, not just texel centres.
"""
    lo, hi = tri.min(1), tri.max(1)
    cell = max(float(np.median(np.max(hi - lo, axis=1))) * 3, 1e-4)
    origin = lo.min(0)
    low = np.floor((lo-origin)/cell).astype(int)
    high = np.floor((hi-origin)/cell).astype(int)
    bins = {}
    for i, (a, b) in enumerate(zip(low, high)):
        for x in range(a[0], b[0]+1):
            for y in range(a[1], b[1]+1):
                bins.setdefault((x,y), []).append(i)
    found = []
    batch = []
    def check(pairs):
        p = np.concatenate(pairs)
        a, b = p.T
        good = np.all(np.minimum(hi[a], hi[b]) - np.maximum(lo[a],lo[b]) > 1e-10, axis=1)
        p = p[good]
        if not len(p): return
        ta, tb = tri[p[:,0]], tri[p[:,1]]
        edges = np.concatenate([np.roll(ta,-1,axis=1)-ta, np.roll(tb,-1,axis=1)-tb],1)
        axes = np.stack([-edges[...,1], edges[...,0]],-1)
        pa = np.einsum('nvc,nac->nav',ta,axes)
        pb = np.einsum('nvc,nac->nav',tb,axes)
        overlap = np.minimum(pa.max(2),pb.max(2))-np.maximum(pa.min(2),pb.min(2))
        good = np.all(overlap > 1e-10*np.linalg.norm(axes,axis=2),axis=1)
        if good.any(): found.append(p[good])
    n = 0
    for ids in bins.values():
        if len(ids)<2: continue
        ii,jj = np.triu_indices(len(ids),1)
        pairs = np.asarray(ids)[np.stack([ii,jj],1)]
        batch.append(pairs); n += len(pairs)
        if n>200000:
            check(batch); batch=[]; n=0
    if batch: check(batch)
    return np.unique(np.concatenate(found),axis=0) if found else np.empty((0,2),int)


def sample(image, uv):
    return cv2.remap(image, (uv[...,0]*image.shape[1]-.5).astype(np.float32),
                     ((1-uv[...,1])*image.shape[0]-.5).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def facing_score(scene, points, normal, d, distance):
    """Per triangle: its facing cosine to d when all sample points see d unoccluded, else -1."""
    hit = cast(scene, points+d*distance, -d)['t_hit'].numpy()
    visible = np.all(np.abs(hit-distance)<2e-5, axis=1)
    return np.where(visible & (normal@d > .08), normal@d, -1.)


def drop_overlaps(owner, specs, v, f, t):
    """Positive-area overlaps within a panel: the farther triangle goes back to the reserve."""
    removed = 0
    for p, (name, d, r, u, box) in enumerate(specs):
        ids = np.flatnonzero(owner==p)
        if not len(ids): continue
        proj = np.stack([v@r, v@u], 1)
        pairs = overlap_pairs(proj[f[ids]])
        if len(pairs):
            depth = t[ids].mean(1)@d
            loser = np.where(depth[pairs[:,0]]<depth[pairs[:,1]], pairs[:,0], pairs[:,1])
            loser = np.unique(loser); owner[ids[loser]] = -1; removed += len(loser)
        print(name, 'overlap pairs', len(pairs), 'owned triangles', int((owner==p).sum()), flush=True)
    return removed


def sphere_directions(n):
    """A Fibonacci sphere, deterministic; the candidate oblique directions."""
    i = np.arange(n)+.5
    phi = np.arccos(1-2*i/n); theta = np.pi*(1+5**.5)*i
    return np.stack([np.cos(theta)*np.sin(phi), np.cos(phi), np.sin(theta)*np.sin(phi)], 1)


def panel_basis(d, front, up):
    """Right/up image axes for a view direction: world up unless looking along it."""
    u0 = up if abs(d@up) < .99 else front*np.sign(d@up)
    u = u0-(u0@d)*d; u /= np.linalg.norm(u)
    return np.cross(u, d), u


def panel_scale(proj, box, pad):
    """px per metre of a projection fitted into a pixel box."""
    x0,y0,x1,y1 = box
    return min((x1-x0-2*pad)/np.ptp(proj[:,0]), (y1-y0-2*pad)/np.ptp(proj[:,1]))


def grid_boxes(n, rows, region):
    """n equal pixel boxes, `rows` rows, filling the pixel region (x0,y0,x1,y1)."""
    cols = -(-n//rows); X0,Y0,X1,Y1 = region; w = (X1-X0)/cols; h = (Y1-Y0)/rows
    return [(round(X0+c*w), round(Y0+r*h), round(X0+(c+1)*w), round(Y0+(r+1)*h)) for r in range(rows) for c in range(cols)][:n]


def pick_panels(cands, movable, own_eff, fit, box, K, prefix, geo):
    """Greedy view selection for one panel group.

cands: candidate directions; movable: triangles this group may take; own_eff: their current
effective density; fit: vertex mask the panel frames (None = whole mesh); box: the pixel box
every panel of the group gets. Returns specs (name,d,r,u,fit) and the panel index per triangle
(-1 = not taken), by the density rule: a view takes a triangle only if it gives it more texels."""
    v,f,t,points,normal,area,scene,distance,front,up,right,pad = geo
    ids = np.flatnonzero(movable)
    if not len(ids) or K<=0: return [], np.full(len(f),-1)
    bases = [panel_basis(d,front,up) for d in cands]
    sel = slice(None) if fit is None else fit
    cscale = np.array([panel_scale(np.stack([v[sel]@r,v[sel]@u],1),box,pad) for r,u in bases])
    cscore = np.stack([facing_score(scene,points[ids],normal[ids],d,distance) for d in cands],1)
    better = (cscore>0)&(cscale[None,:]**2*cscore>own_eff[ids][:,None])
    covered = np.zeros(len(ids),bool); picks = []
    for k in range(K):
        gain = ((better&~covered[:,None])*area[ids][:,None]).sum(0); j = int(gain.argmax())
        if gain[j]<=0: break
        covered |= better[:,j]; picks.append(j)
        print(prefix,'panel',k,'direction',np.round(cands[j],3),'adds area',float(gain[j]/area.sum()),flush=True)
    specs = []
    for k,j in enumerate(picks):
        d = cands[j]; r,u = bases[j]; el = np.degrees(np.arcsin(d@up)); az = np.degrees(np.arctan2(d@right,d@front))
        specs.append((f'{prefix}{k}_az{az:+.0f}_el{el:+.0f}',d,r,u,fit))
    sub = np.where(better[:,picks],cscore[:,picks],-1.); owner = np.full(len(f),-1)
    take = sub.max(1)>0; owner[ids[take]] = sub.argmax(1)[take]
    return specs, owner


def build_sheet(out, W, H, pad, specs, owner, fallback, mesh, olduv, source, scene, centre, distance, protect_sources, extra_stats):
    """Lay the owned triangles of each panel into its pixel box, render the context, rasterize the sheet.

specs: (name, d, r, u, fit, box). With `fallback` (a pixel box) the unowned triangles keep their source
UVs inside it and `mesh_uv.obj` addresses the whole mesh; without it only owned triangles are rasterized."""
    out.mkdir(parents=True, exist_ok=True)
    v = np.asarray(mesh.vertices); f = np.asarray(mesh.triangles); t = v[f]
    cross = np.cross(t[:,1]-t[:,0], t[:,2]-t[:,0]); area = np.linalg.norm(cross, axis=1)/2
    uv = np.zeros_like(olduv); context = np.full((H,W,3),180,np.uint8); seen = np.zeros((H,W),bool)
    panels = []
    for p, (name, d, r, u, fit, box) in enumerate(specs):
        proj = np.stack([v@r, v@u], 1); sel = proj if fit is None else proj[fit]
        x0,y0,x1,y1 = box; w = x1-x0; h = y1-y0
        bounds = np.stack([sel.min(0), sel.max(0)]); mid = bounds.mean(0)
        scale = min((w-2*pad)/np.ptp(sel[:,0]), (h-2*pad)/np.ptp(sel[:,1]))
        pix = (proj-mid)*[scale,-scale]+[(x0+x1)/2,(y0+y1)/2]
        ids = owner==p; uv[ids] = pix[f[ids]]/[W,H]; uv[ids,:,1] = 1-uv[ids,:,1]
        xx,yy = np.meshgrid(np.arange(x0,x1)+.5, np.arange(y0,y1)+.5)
        pu = (xx-(x0+x1)/2)/scale+mid[0]; pv = (-(yy-(y0+y1)/2))/scale+mid[1]
        origins = pu[...,None]*r+pv[...,None]*u+(centre@d+distance)*d
        hit = cast(scene, origins, -d); valid = np.isfinite(hit['t_hit'].numpy()); pid = hit['primitive_ids'].numpy(); pid[~valid] = 0
        bary = hit['primitive_uvs'].numpy(); bw = np.stack([1-bary.sum(-1), bary[...,0], bary[...,1]], -1)
        suv = (olduv[pid]*bw[...,None]).sum(-2)
        colors = sample(source, suv); context[y0:y1,x0:x1][valid] = colors[valid]; seen[y0:y1,x0:x1] = valid
        panels.append(dict(name=name, box_pixels=[int(x0),int(y0),int(x1),int(y1)], direction=[float(z) for z in d], px_per_m=float(scale),
                           head=fit is not None, triangles=int(ids.sum()), area_fraction=float(area[ids].sum()/area.sum())))
    if fallback is not None:
        # Preserve the old chart layout for every face that cannot safely use a view.
        x0,y0,x1,y1 = fallback
        uv[owner<0] = olduv[owner<0]*[(x1-x0-2*pad)/W,(y1-y0-2*pad)/H]+[(x0+pad)/W,(H-y1+pad)/H]
        panels.append(dict(name='fallback', box_pixels=[int(x0),int(y0),int(x1),int(y1)], triangles=int((owner<0).sum()), area_fraction=float(area[owner<0].sum()/area.sum())))
        tri_ids = np.arange(len(f))
    else:
        tri_ids = np.flatnonzero(owner>=0)
    # Rasterize the new UVs, transferring existing pixels via original barycentrics.
    flat = np.concatenate([uv[tri_ids].reshape(-1,2), np.zeros((len(tri_ids)*3,1))], 1)
    atlas_scene = scene_for(flat, np.arange(len(tri_ids)*3).reshape(-1,3))
    texture = context.copy(); mask = np.zeros((H,W),np.uint8); remap = np.full((H,W,2),-1,np.float32)
    position = np.memmap(out/'position.f32', mode='w+', dtype=np.float32, shape=(H,W,3))
    normals = np.memmap(out/'normal.f32', mode='w+', dtype=np.float32, shape=(H,W,3))
    mesh.compute_vertex_normals(); vn = np.asarray(mesh.vertex_normals)
    for y in range(0,H,128):
        yy,xx = np.meshgrid(np.arange(y,min(y+128,H)), np.arange(W), indexing='ij')
        origins = np.stack([(xx+.5)/W, 1-(yy+.5)/H, np.ones_like(xx)], -1)
        hit = cast(atlas_scene, origins, [0,0,-1]); valid = np.isfinite(hit['t_hit'].numpy()); pid = hit['primitive_ids'].numpy(); pid[~valid] = 0
        pid = tri_ids[pid]
        b = hit['primitive_uvs'].numpy(); bw = np.stack([1-b.sum(-1), b[...,0], b[...,1]], -1)
        suv = (olduv[pid]*bw[...,None]).sum(-2); color = sample(source, suv)
        sl = slice(y,y+len(yy)); texture[sl][valid] = color[valid]; mask[sl] = valid*255; remap[sl][valid] = suv[valid]
        pos = (t[pid]*bw[...,None]).sum(-2); nn = (vn[f[pid]]*bw[...,None]).sum(-2); nn /= np.maximum(np.linalg.norm(nn,axis=-1,keepdims=True),1e-20)
        position[sl] = np.where(valid[...,None],pos,0); normals[sl] = np.where(valid[...,None],nn,0)
    position.flush(); normals.flush()
    # Give gutters outside the silhouette the owned surface color; inside it the view context
    # already IS the surface (on the sparse side panels a gutter over context stamps discs).
    dist,labels = cv2.distanceTransformWithLabels(255-mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(labels.max()+1,np.int64); lut[labels[mask>0]] = np.flatnonzero(mask)
    nearest = lut[labels]; gutter = (mask==0)&(dist<=pad)&~seen
    texture[gutter] = texture.reshape(-1,3)[nearest[gutter]]
    cv2.imwrite(str(out/'texture.png'), texture)
    cv2.imwrite(str(out/'context.png'), context)
    cv2.imwrite(str(out/'mask.png'), mask)
    cv2.imwrite(str(out/'mask_dilated.png'), ((mask>0)|gutter).astype(np.uint8)*255)
    np.save(out/'source_uv.npy', remap)
    np.save(out/'face_panel.npy', owner.astype(np.int8))
    np.save(out/'face_uv.npy', uv)
    diffusion = texture.copy(); edit_mask = mask.copy()
    if fallback is not None:
        diffusion[y0:y1,x0:x1] = 180; edit_mask[y0:y1,x0:x1] = 0
        cv2.imwrite(str(out/'reserve_texture.png'), source)
    cv2.imwrite(str(out/'diffusion_texture.png'), diffusion)
    cv2.imwrite(str(out/'edit_mask.png'), edit_mask)
    for path in protect_sources:
        src = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        o = sample(src, remap); o[mask==0] = 0
        cv2.imwrite(str(out/path.name), o)
    if fallback is not None:
        with (out/'mesh_uv.obj').open('w') as obj:
            obj.write('mtllib mesh_uv.mtl\n')
            for p in v: obj.write('v %.9g %.9g %.9g\n'%tuple(p))
            for p in uv.reshape(-1,2): obj.write('vt %.9g %.9g\n'%tuple(p))
            obj.write('usemtl tex\n')
            for i,face in enumerate(f): obj.write('f '+' '.join(f'{vi+1}/{i*3+j+1}' for j,vi in enumerate(face))+'\n')
        (out/'mesh_uv.mtl').write_text('newmtl tex\nKd 1 1 1\nmap_Kd texture.png\n')
    stats = dict(width=W, height=H, pad=pad, verts=len(v), tris=len(f), uv_verts=len(f)*3, covered=int((mask>0).sum()), panels=panels, **extra_stats,
                 files=dict(mesh='mesh_uv.obj' if fallback is not None else None, texture='texture.png', position='position.f32', normal='normal.f32', mask='mask.png', mask_dilated='mask_dilated.png'))
    if W==H: stats['res'] = W
    (out/'atlas.json').write_text(json.dumps(stats, indent=2)+'\n')
    cv2.imwrite(str(out/'preview.jpg'), cv2.resize(diffusion, (1024*W//max(W,H),1024*H//max(W,H)), interpolation=cv2.INTER_AREA))
    return stats, uv


def write_layered_obj(out, v, f, olduv, layers):
    """One OBJ, one material per sheet plus the reserve at source UVs, so a viewer shows every layer."""
    with (out/'mesh_diffusion.obj').open('w') as obj:
        obj.write('mtllib mesh_diffusion.mtl\n')
        for p in v: obj.write('v %.9g %.9g %.9g\n'%tuple(p))
        direct_uv = olduv.copy()
        for name, owner, uv, image in layers: direct_uv[owner>=0] = uv[owner>=0]
        for p in direct_uv.reshape(-1,2): obj.write('vt %.9g %.9g\n'%tuple(p))
        unowned = np.ones(len(f), bool)
        for name, owner, uv, image in layers:
            obj.write('usemtl '+name+'\n')
            for i in np.flatnonzero(owner>=0):
                obj.write('f '+' '.join(f'{vi+1}/{i*3+j+1}' for j,vi in enumerate(f[i]))+'\n')
            unowned &= owner<0
        obj.write('usemtl reserve\n')
        for i in np.flatnonzero(unowned):
            obj.write('f '+' '.join(f'{vi+1}/{i*3+j+1}' for j,vi in enumerate(f[i]))+'\n')
    (out/'mesh_diffusion.mtl').write_text(''.join(f'newmtl {name}\nKd 1 1 1\nmap_Kd {image}\n' for name,_,_,image in layers)
                                          +'newmtl reserve\nKd 1 1 1\nmap_Kd reserve_texture.png\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--atlas', type=Path, required=True)
    ap.add_argument('--texture',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--res',type=int,default=4096,help='Sheet size in multi layout (square)')
    ap.add_argument('--front',type=float,default=0,help='Yaw degrees: 0 looks toward +Z, Y is up')
    ap.add_argument('--front-min-cos',type=float,default=.25)
    ap.add_argument('--pad',type=int,default=8)
    ap.add_argument('--layout',choices=['multi','single'],default='multi',help="'multi': one sheet per group (main, extra/, head/); 'single': every panel on one wide sheet")
    ap.add_argument('--single-size',default='5376x3072',help='WxH of the single sheet')
    ap.add_argument('--single-split',type=float,default=.58,help='Single sheet: fraction of the height above which the oblique panels sit, the head panels below')
    ap.add_argument('--extra',type=int,default=6,help='Oblique full-body panels (0 = none)')
    ap.add_argument('--extra-candidates',type=int,default=64,help='Candidate directions the extra/head panels are chosen from')
    ap.add_argument('--extra-scope',choices=['reserve','small','grazing'],default='grazing',help="'reserve': only the reserve may move to the extra panels; 'small': also everything the low-density left/right/top/bottom panels own; 'grazing': also front/back triangles facing their panel below --steal-cos")
    ap.add_argument('--steal-cos',type=float,default=.5,help='grazing scope: front/back triangles facing their panel below this cosine may move')
    ap.add_argument('--head',type=int,default=4,help='Close-up head panels (0 = none)')
    ap.add_argument('--head-height',type=float,default=.32,help='The head band: metres below the crown that head panels frame and may take')
    a=ap.parse_args()
    if a.res<512 or not 0<a.front_min_cos<1: ap.error('res >= 512 and 0 < front-min-cos < 1 required')
    if not 0 <= a.pad < a.res*.03: ap.error('pad must be nonnegative and less than 3% of res')
    if a.output.resolve()==a.atlas.resolve(): ap.error('Output must differ from the source atlas')
    if a.extra<0 or a.extra>12 or a.head<0 or a.head>8 or a.extra_candidates<max(a.extra,a.head): ap.error('0 <= extra <= 12, 0 <= head <= 8 and extra-candidates >= both required')
    a.output.mkdir(parents=True,exist_ok=True)
    mesh=o3d.io.read_triangle_mesh(str(a.atlas/'mesh_uv.obj'),enable_post_processing=False)
    v=np.asarray(mesh.vertices); f=np.asarray(mesh.triangles); olduv=np.asarray(mesh.triangle_uvs).reshape(-1,3,2)
    if len(olduv)!=len(f): raise ValueError('Input needs triangle UVs')
    source_texture=a.texture or a.atlas/'texture.png'
    source=cv2.imread(str(source_texture))
    if source is None: raise ValueError('Cannot read source texture')
    t=v[f]; cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]); area=np.linalg.norm(cross,axis=1)/2
    normal=cross/np.maximum(2*area[:,None],1e-20)
    yaw=np.deg2rad(a.front); front=np.array([np.sin(yaw),0,np.cos(yaw)]); up=np.array([0.,1,0]); right=np.cross(up,front)
    scene=scene_for(v,f); centre=(v.min(0)+v.max(0))/2; distance=np.linalg.norm(np.ptp(v,axis=0))*2
    weights=np.array([[1/3]*3,[.8,.1,.1],[.1,.8,.1],[.1,.1,.8],[.49,.49,.02],[.02,.49,.49],[.49,.02,.49]])
    points=np.einsum('sk,nkc->nsc',weights,t)
    geo=(v,f,t,points,normal,area,scene,distance,front,up,right,a.pad)
    # Sheet geometry per layout: the six axis panels, the oblique grid, the head grid.
    if a.layout=='multi':
        R=a.res; main_boxes=[tuple(round(z*R) for z in b) for b in [(0,0,.5,.72),(.5,0,1,.72),(0,.72,.25,1),(.25,.72,.5,1),(.5,.72,.75,.86),(.5,.86,.75,1)]]
        fallback=tuple(int(z) for z in np.rint(np.array([.75,.75,1.,1.])*R)); main_size=(R,R)
        extra_boxes=grid_boxes(a.extra,2 if a.extra>3 else 1,(0,0,R,R)); head_boxes=grid_boxes(a.head,2 if a.head>2 else 1,(0,0,R,R))
    else:
        W,H=[int(z) for z in a.single_size.lower().split('x')]; fw=round(.225*W); ymid=round(a.single_split*H)
        main_boxes=[(0,0,fw,H),(fw,0,2*fw,H)]+[None]*4; fallback=None; main_size=(W,H)
        extra_boxes=grid_boxes(a.extra,2 if a.extra>3 else 1,(2*fw,0,W,ymid)); head_boxes=grid_boxes(a.head,1,(2*fw,ymid,W,H))
    axes=[('front',front,right,up),('back',-front,-right,up),('left',right,-front,up),('right',-right,front,up),('top',up,right,-front),('bottom',-up,right,front)]
    scores=np.stack([facing_score(scene,points,normal,d,distance) for name,d,r,u in axes],1)
    for (name,d,r,u),sc in zip(axes,scores.T): print(name,'visible area',float(area[sc>0].sum()/area.sum()),flush=True)
    owner=scores.argmax(1); owner[scores.max(1)<0]=-1
    for p in (0,1): owner[scores[:,p]>=a.front_min_cos]=p
    if a.layout=='single': owner[owner>=2]=-1  # no side/top/bottom panels on the single sheet: the obliques take them
    main_specs=[(name,d,r,u,None,box) for (name,d,r,u),box in zip(axes,main_boxes) if box is not None]
    overlap_removed=drop_overlaps(owner,[(n,d,r,u,None) for n,d,r,u,_,_ in main_specs],v,f,t)
    scale_main=np.array([panel_scale(np.stack([v@r,v@u],1),box,a.pad) for n,d,r,u,_,box in main_specs])
    own_cos=np.where(owner>=0,scores[np.arange(len(f)),np.maximum(owner,0)],-1.)
    own_eff=np.where(owner>=0,scale_main[np.maximum(owner,0)]**2*np.maximum(own_cos,0),0.)
    cands=np.concatenate([np.array([d for n,d,r,u in axes[2:]],float),sphere_directions(a.extra_candidates)])
    groups=[]  # (group name, specs, owner array)
    if a.extra:
        movable=(owner<0) if a.extra_scope=='reserve' else (owner!=0)&(owner!=1)
        if a.extra_scope=='grazing': movable|=own_cos<a.steal_cos
        specs_x,owner_x=pick_panels(cands,movable,own_eff,None,extra_boxes[0],a.extra,'e',geo)
        specs_x=[(n,d,r,u,fit,box) for (n,d,r,u,fit),box in zip(specs_x,extra_boxes)]
        drop_overlaps(owner_x,[(n,d,r,u,None) for n,d,r,u,_,_ in specs_x],v,f,t)
        taken=owner_x>=0; owner[taken]=-1
        sx=np.array([panel_scale(np.stack([v@r,v@u],1),box,a.pad) for n,d,r,u,_,box in specs_x]) if specs_x else np.zeros(0)
        own_eff[taken]=sx[owner_x[taken]]**2*np.array([normal[i]@specs_x[k][1] for i,k in zip(np.flatnonzero(taken),owner_x[taken])])
        groups.append(('extra',specs_x,owner_x))
    if a.head:
        # The head band: everything within --head-height of the crown; the head panels frame only it.
        band_v=v[:,1]>v[:,1].max()-a.head_height; band_t=t[:,:,1].mean(1)>v[:,1].max()-a.head_height
        specs_h,owner_h=pick_panels(cands,band_t,own_eff,band_v,head_boxes[0],a.head,'h',geo)
        specs_h=[(n,d,r,u,fit,box) for (n,d,r,u,fit),box in zip(specs_h,head_boxes)]
        drop_overlaps(owner_h,[(n,d,r,u,None) for n,d,r,u,_,_ in specs_h],v,f,t)
        taken=owner_h>=0; owner[taken]=-1
        for g in groups: g[2][taken]=-1
        groups.append(('head',specs_h,owner_h))
    protect_sources=sorted(a.atlas.glob('protect*.png'))
    common=dict(source=str(a.atlas),source_texture=str(source_texture),front_yaw=a.front,layout=a.layout)
    layers=[]
    if a.layout=='multi':
        stats,uv=build_sheet(a.output,*main_size,a.pad,main_specs,owner,fallback,mesh,olduv,source,scene,centre,distance,protect_sources,dict(overlap_removed=overlap_removed,**common))
        layers.append(('diffusion',owner,uv,'diffusion_texture.png'))
        for gname,specs_g,owner_g in groups:
            gstats,guv=build_sheet(a.output/gname,*main_size,a.pad,specs_g,owner_g,None,mesh,olduv,source,scene,centre,distance,protect_sources,dict(parent=str(a.output),group=gname,**common))
            stats[gname]=dict(panels=gstats['panels'],owned_area_fraction=float(area[owner_g>=0].sum()/area.sum()))
            layers.append((gname,owner_g,guv,f'{gname}/diffusion_texture.png'))
        (a.output/'atlas.json').write_text(json.dumps(stats,indent=2)+'\n')
    else:
        specs=list(main_specs); combined=owner.copy(); combined[owner>=0]=owner[owner>=0]
        for gname,specs_g,owner_g in groups:
            base=len(specs); combined[owner_g>=0]=base+owner_g[owner_g>=0]; specs+=specs_g
        stats,uv=build_sheet(a.output,*main_size,a.pad,specs,combined,None,mesh,olduv,source,scene,centre,distance,protect_sources,dict(overlap_removed=overlap_removed,**common))
        layers.append(('diffusion',combined,uv,'diffusion_texture.png'))
        cv2.imwrite(str(a.output/'reserve_texture.png'),source)
    # One OBJ carries every sheet; hidden geometry keeps the original texture resolution.
    write_layered_obj(a.output,v,f,olduv,layers)
    print(json.dumps({k:vv for k,vv in stats.items() if k!='panels'},indent=2),flush=True)
    for p in stats['panels']: print(p['name'],p['triangles'],'area %.4f'%p['area_fraction'],'px/m %.0f'%p.get('px_per_m',0),flush=True)


if __name__=='__main__': main()
