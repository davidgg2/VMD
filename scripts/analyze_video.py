"""Run released models on every frame; report descriptive, non-GT diagnostics."""
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'external/DA-W'))
from infer import load_daw, load_dav2, build_transform, run_forward
sys.path.insert(0, str(ROOT/'external/object-concepts-from-motion/tools'))
from pytorch_model import FlowSegModel

def load_model(checkpoint_path, arch, device):
    # Same strict upstream loading; avoids importing its optional sklearn demo.
    checkpoint=torch.load(checkpoint_path,map_location='cpu',mmap=True,weights_only=True)
    state=checkpoint.get('state_dict',checkpoint)
    if 'logit_scale' in state:
        state=state.copy(); state.pop('logit_scale')
    with torch.device('meta'): model=FlowSegModel(arch=arch)
    model.load_state_dict(state,strict=True,assign=True)
    return model.to(device).eval()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--video', default=r'D:\bsense\movies\night_20260511_111600.mp4')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--output', default='outputs/analysis')
    args=p.parse_args()
    out=ROOT/args.output; out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    daw,style=load_daw(ROOT/'checkpoints/daw_vits_stage2.pth',ROOT/'checkpoints/daw_style_filter_stage1.pth',device)
    dav,_=load_dav2(ROOT/'checkpoints/depth_anything_v2_vits.pth',device)
    ocm=load_model(ROOT/'checkpoints/swin_t.pth',arch='tiny',device=device)
    transform=build_transform(518)
    cap=cv2.VideoCapture(args.video)
    fps=cap.get(cv2.CAP_PROP_FPS); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w,h=int(cap.get(3)),int(cap.get(4))
    if not cap.isOpened() or fps<=0 or total<=0:
        raise RuntimeError(f'Cannot read video metadata: {args.video}')
    count=min(total,args.limit) if args.limit else total
    writer=cv2.VideoWriter(str(out/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(960,576))
    if not writer.isOpened(): raise RuntimeError('Video writer failed')
    rows=[]; prev_gray=None; prev_features=None; prev_norm={}
    depths={name:np.lib.format.open_memmap(out/f'{name}_relative_disparity.npy',mode='w+',dtype=np.float16,shape=(count,128,160)) for name in ('daw','dav2')}
    flow_sum=np.zeros((256,320),np.float64); basis=None; ranges={}
    start=time.time()
    valid=np.ones((256,320),bool); valid[:14]=False; valid[120:134,148:163]=False
    def panel(im,title):
        im=cv2.resize(im,(320,256)); bar=np.zeros((32,320,3),np.uint8)
        cv2.putText(bar,title,(6,22),0,.48,(240,240,240),1,cv2.LINE_AA)
        return np.vstack([bar,im])
    for idx in range(total if not args.limit else min(total,args.limit)):
        ok,bgr=cap.read()
        if not ok: break
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        gray=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),(320,256))
        row={'frame':idx,'seconds':idx/fps,'luma_mean':float(gray[valid].mean()),'laplacian_variance':float(cv2.Laplacian(gray,cv2.CV_32F)[valid].var())}
        residual=np.zeros((256,320),np.float32)
        if prev_gray is not None:
            flow=cv2.calcOpticalFlowFarneback(prev_gray,gray,None,.5,3,21,3,5,1.2,0)
            global_xy=np.median(flow[valid],axis=0)
            residual=np.linalg.norm(flow-global_xy,axis=2)
            flow_sum+=residual
            row.update(global_translation_px=float(np.linalg.norm(global_xy)*w/320),residual_flow_median_px=float(np.median(residual[valid])*w/320),residual_flow_p95_px=float(np.percentile(residual[valid],95)*w/320))
        tensor=torch.from_numpy(transform({'image':rgb.astype(np.float32)/255})['image']).unsqueeze(0).to(device)
        colors={}
        with torch.inference_mode():
            for name,model,sf in [('daw',daw,style),('dav2',dav,None)]:
                pred=run_forward(model,tensor,sf)
                d=F.interpolate(pred.reshape(1,1,*pred.shape[-2:]),size=(128,160),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
                if not np.isfinite(d).all(): raise RuntimeError(f'Nonfinite {name} frame {idx}')
                depths[name][idx]=d
                lo,hi=np.percentile(d,[2,98]); norm=(d-lo)/max(hi-lo,1e-6)
                metric_mask=np.ones(d.shape,bool); metric_mask[:7]=False; metric_mask[59:68,73:83]=False
                if name in prev_norm: row[name+'_temporal_mae']=float(np.abs(norm-prev_norm[name])[metric_mask].mean())
                prev_norm[name]=norm
                if name not in ranges: ranges[name]=(float(lo),float(hi))
                a,b=ranges[name]
                colors[name]=cv2.applyColorMap(np.uint8(np.clip((d-a)/max(b-a,1e-6),0,1)*255),cv2.COLORMAP_INFERNO)
            # Native aspect and full field of view; upstream RGB normalization.
            x=torch.from_numpy((rgb.astype(np.float32)-[123.675,116.28,103.53])/[58.395,57.12,57.375]).permute(2,0,1).unsqueeze(0).float().to(device)
            feat=F.normalize(ocm(x).float(),dim=1)
            small=F.normalize(F.interpolate(feat,size=(32,40),mode='bilinear',align_corners=False),dim=1)[0].cpu().numpy()
        flat=small.reshape(small.shape[0],-1).T
        if basis is None:
            center=flat.mean(0); _,_,vt=np.linalg.svd(flat-center,full_matrices=False); basis=vt[:3].T
            projected=(flat-center)@basis; pc_lo=np.percentile(projected,1,axis=0); pc_hi=np.percentile(projected,99,axis=0)
            np.savez(out/'feature_pca_basis.npz',center=center,basis=basis,lo=pc_lo,hi=pc_hi)
        projected=(flat-center)@basis
        feature_rgb=np.uint8(np.clip((projected-pc_lo)/np.maximum(pc_hi-pc_lo,1e-6),0,1).reshape(32,40,3)*255)
        feature_bgr=cv2.cvtColor(feature_rgb,cv2.COLOR_RGB2BGR)
        change=np.zeros((32,40),np.float32)
        if prev_features is not None:
            change=np.maximum(0,1-(small*prev_features).sum(0))
            row['feature_cosine_change']=float(change[2:30,2:38].mean())
        change_color=cv2.applyColorMap(np.uint8(np.clip(change/.15,0,1)*255),cv2.COLORMAP_MAGMA)
        flow_color=cv2.applyColorMap(np.uint8(np.clip(residual/2,0,1)*255),cv2.COLORMAP_TURBO)
        canvas=np.vstack([np.hstack([panel(bgr,f'Input | {idx/fps:.2f}s'),panel(colors['daw'],'DA-W | relative disparity'),panel(colors['dav2'],'Depth Anything V2 | baseline')]),np.hstack([panel(feature_bgr,'Motion-trained features | fixed PCA'),panel(flow_color,'Residual flow | 0 to 4 native px'),panel(change_color,'Feature change | 0 to 0.15')])])
        writer.write(canvas)
        if idx%125==0 or idx==total-1: cv2.imwrite(str(out/f'comparison_{idx:04d}.jpg'),canvas)
        rows.append(row); prev_gray=gray; prev_features=small
        if idx%250==0:
            for ds in depths.values(): ds.flush()
            progress={'processed_frames':idx+1,'expected_frames':count,'elapsed_seconds':time.time()-start,'video_seconds':idx/fps}
            (out/'progress.json').write_text(json.dumps(progress))
            print(f'{idx+1}/{count} frames; elapsed {time.time()-start:.1f}s',flush=True)
    cap.release(); writer.release()
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with open(out/'metrics.csv','w',newline='') as f:
        writer_csv=csv.DictWriter(f,keys); writer_csv.writeheader(); writer_csv.writerows(rows)
    for ds in depths.values(): ds.flush()
    if len(rows)!=count: raise RuntimeError(f'Decode stopped early: {len(rows)}/{count} frames')
    np.save(out/'mean_residual_flow_320x256.npy',flow_sum/max(len(rows)-1,1))
    summary={k:{'median':float(np.median([r[k] for r in rows if k in r])),'p95':float(np.percentile([r[k] for r in rows if k in r],95))} for k in keys if k not in ('frame','seconds')}
    metadata={'video':args.video,'native_size':[w,h],'fps':fps,'expected_frames':total,'processed_frames':len(rows),'duration_seconds':len(rows)/fps,'elapsed_seconds':time.time()-start,'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,'summary':summary,'visualization_ranges_from_first_frame':ranges,'checkpoints':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'checkpoints').glob('*.pth')},'limitations':['Relative disparity is not metric depth.','No ground truth: temporal consistency is not depth accuracy.','Residual flow includes object motion, camera motion not captured by translation, noise and turbulence.','Feature colors are not semantic classes or instance masks.','Temporal metrics compare fixed pixels, not motion-compensated correspondences.','Input sensor modality is unknown.'],'configuration':{'daw_resize':518,'ocm_input':'native full-frame RGB','depth_storage':'160x128 float16','flow':'Farneback 320x256, global median translation removed','feature_pca':'fixed first-frame basis','models':'DA-W ViT-S; DAV2 ViT-S; OCM Swin-T'}}
    (out/'summary.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__': main()
