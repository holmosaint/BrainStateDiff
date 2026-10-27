"""Generate matched-state or cross-state responses from training histories."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from .train import seed_all, build_models, build_scheduler, load_weights
from .data import ImageEmbeddingStore
from .latent import LatentStats, encode_frames_with_vae, decode_latent_video_tensor
from .sampling import guided_ddim_sample_target


def conditioning_pairs(keys, counts, target_key, mode):
    """Matched empirical pairs or the Cartesian product with the empirical state pool."""
    if mode not in {"matched_state", "cross_state"}:
        raise ValueError("mode must be matched_state or cross_state")
    donors = [target_key] if mode == "matched_state" else keys
    return [(key, trial) for key in donors for trial in range(counts[key])]


def validate_checkpoint_config(config, checkpoint, state_key="model_state_dict"):
    """Check subject-token identity and the diffusion schedule before generation."""
    training = config["diffusion"].get("training", {})
    supplied = dict(prediction_type=training.get("prediction_type", "x0"),
                    diffusion_steps=training.get("diffusion_steps", 1000),
                    beta_schedule=training.get("beta_schedule", "squaredcos_cap_v2"))
    if supplied["prediction_type"] not in {"x0", "epsilon"}:
        raise ValueError("prediction_type must be x0 or epsilon")
    if state_key in {"model_state_dict", "ema_state_dict"}:
        saved = checkpoint["config"]
        names = [subject["name"] for subject in saved["subjects"]]
        saved_training = saved["diffusion"].get("training", {})
        expected = dict(prediction_type=saved_training.get("prediction_type", "x0"),
                        diffusion_steps=saved_training.get("diffusion_steps", 1000),
                        beta_schedule=saved_training.get("beta_schedule", "squaredcos_cap_v2"))
    elif state_key in {"model_state", "ema_state"}:
        names = checkpoint["subject_names"]
        objective = checkpoint["prediction_type"]
        scheduler_prediction = "sample" if objective == "x0" else "epsilon"
        if checkpoint["scheduler_prediction_type"] != scheduler_prediction:
            raise ValueError("Original checkpoint objective and scheduler prediction type disagree")
        # The original trainer did not save the timestep count or beta schedule.
        if not {"diffusion_steps", "beta_schedule"}.issubset(training):
            raise ValueError("Original checkpoints require explicit diffusion_steps and beta_schedule in the config")
        expected = dict(supplied, prediction_type=objective)
    else:
        raise ValueError("Use model_state_dict/ema_state_dict for local checkpoints or model_state/ema_state for original checkpoints")
    if [subject["name"] for subject in config["subjects"]] != names:
        raise ValueError("Subject names and order must match the training checkpoint's embedding rows")
    if supplied != expected:
        raise ValueError(f"Diffusion schedule differs from training checkpoint: expected {expected}, got {supplied}")


def validate_generation_window(window):
    """Require the full onset history and response to lie within the configured crop."""
    history=window["history_frames"];onset=window["stim_onset_frame"]
    if history<1 or onset<history:
        raise ValueError("Generation requires 0 < history_frames <= stimulus onset")
    first=onset-history;last=onset+window["target_frames"]
    end=window.get("frame_end")
    if window.get("frame_st",0)>first or (end is not None and end<last):
        raise ValueError(f"Generation window must contain the full onset history and response [{first}, {last})")


@torch.no_grad()
def generate_subject(model, vae, scheduler, stats, store, real_h5, output_h5, subject_idx,
                     window, mode="matched_state", replicas=1, batch_size=8, steps=50,
                     cfg_scale=1., device="cpu", seed=42):
    """Emit responses in source order: seed+replica, then all image/history pairs."""
    if replicas<1 or batch_size<1: raise ValueError("replicas and batch_size must be positive")
    validate_generation_window(window)
    device=torch.device(device)
    history_frames=window["history_frames"]; target_frames=window["target_frames"]
    onset=window["stim_onset_frame"]
    cond_start=window.get("cond_start_frame")
    if cond_start is None: cond_start=onset
    mask=(torch.arange(onset-history_frames,onset+target_frames,device=device)>=cond_start).float()
    model.eval();vae.eval()
    with h5py.File(real_h5,"r") as real,h5py.File(output_h5,"x") as out:
        keys=sorted(k for k in real if k!="mask")
        counts={k:real[k].shape[0] for k in keys}
        out.attrs["sampling_mode"]=mode
        out.attrs["subject_idx"]=subject_idx
        out.attrs["history_source"]=str(real_h5)
        out.attrs["base_seed"]=seed
        work_items=[];sizes={}
        for target_key in keys:
            pairs=conditioning_pairs(keys,counts,target_key,mode)
            sizes[target_key]=len(pairs)
            group=out.create_group(target_key)
            group.create_dataset("source_key",data=np.asarray([k for k,_ in pairs]*replicas,dtype=h5py.string_dtype()))
            group.create_dataset("source_trial_idx",data=[i for _,i in pairs]*replicas)
            group.create_dataset("replica_idx",data=np.repeat(np.arange(replicas),len(pairs)))
            group.attrs["num_samples"]=len(pairs)*replicas
            work_items.extend((target_key,k,i,pos) for pos,(k,i) in enumerate(pairs))
        for replica in range(replicas):
            # Reseed after model construction, separately for every subject/replica.
            seed_all(seed+replica)
            for start in range(0,len(work_items),batch_size):
                chunk=work_items[start:start+batch_size];b=len(chunk)
                history=np.stack([real[k][i,onset-history_frames:onset] for _,k,i,_ in chunk])
                if history.ndim==4: history=history[:,:,None]
                hist=torch.from_numpy(history.astype(np.float32)).to(device)
                empty=torch.zeros(b,target_frames,*hist.shape[2:],device=device)
                history_vol,_=encode_frames_with_vae(vae,hist,empty,stats)
                embedding=torch.from_numpy(np.stack([store.get(key) for key,_,_,_ in chunk])).to(device)
                sampled=guided_ddim_sample_target(model,scheduler,history_vol,embedding,
                    mask[None].expand(b,-1),torch.full((b,),subject_idx,dtype=torch.long,device=device),
                    target_frames,steps,device,cfg_scale,0.,0.)
                frames=decode_latent_video_tensor(vae,sampled,stats).cpu().numpy()
                if frames.shape[2]!=1:raise ValueError("Widefield output must be single-channel")
                frames=frames[:,:,0]
                for row,(key,_,_,pos) in enumerate(chunk):
                    group=out[key]
                    if "samples" not in group:
                        group.create_dataset("samples",shape=(sizes[key]*replicas,*frames.shape[1:]),dtype="float32",chunks=True)
                    group["samples"][replica*sizes[key]+pos]=frames[row]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True);p.add_argument("--checkpoint",required=True)
    p.add_argument("--latent-stats",help="Separate original mean/std/count checkpoint; new checkpoints embed these")
    p.add_argument("--state-key",default="model_state_dict",help="Checkpoint field; use ema_state_dict for EMA weights")
    p.add_argument("--output",required=True);p.add_argument("--mode",choices=["matched_state","cross_state"],required=True)
    p.add_argument("--replicas",type=int,default=1);p.add_argument("--steps",type=int,default=50)
    p.add_argument("--batch-size",type=int,default=8);p.add_argument("--cfg-scale",type=float,default=1.)
    p.add_argument("--device",default="cpu");p.add_argument("--seed",type=int,default=42)
    a=p.parse_args();config=json.loads(Path(a.config).read_text())
    store=ImageEmbeddingStore(config["image_embeddings"],config.get("image_emb_dim",768))
    vae,model=build_models(config,store,a.device)
    checkpoint=load_weights(model,a.checkpoint,a.state_key,a.device)
    validate_checkpoint_config(config,checkpoint,a.state_key)
    raw=torch.load(a.latent_stats,map_location="cpu",weights_only=True) if a.latent_stats else checkpoint["latent_stats"]
    stats=LatentStats(raw["mean"],raw["std"],raw["count"]).to(torch.device(a.device))
    scheduler=build_scheduler(config["diffusion"].get("training",{}))
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    for index,subject in enumerate(config["subjects"]):
        generate_subject(model,vae,scheduler,stats,store,subject["train"],output/f'{subject["name"]}_{a.mode}.h5',
            index,config["diffusion"]["window"],a.mode,a.replicas,a.batch_size,a.steps,a.cfg_scale,a.device,a.seed)

if __name__=="__main__":main()
