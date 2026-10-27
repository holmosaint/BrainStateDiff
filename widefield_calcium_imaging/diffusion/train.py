"""Train the shared frame VAE or the state-conditioned latent diffusion model."""
from __future__ import annotations
import argparse
import copy
import json
import logging
import random
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from diffusers import DDIMScheduler
from .vae import UNet2D, BetaScheduler
from . import vae_engine, engine
from .data import ImageEmbeddingStore, SlidingWindowLatentDataset, MultiSubjectSlidingWindowDataset, BalancedMultiSubjectBatchSampler, collate_fn
from .model import MultiSubjectSlidingWindowModel
from .latent import LatentStats, TrainState, EMA, encode_frames_with_vae


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_weights(model, path, key="model_state_dict", device="cpu"):
    """Load a documented checkpoint field; DDP prefixes are removed explicitly."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    weights = checkpoint[key]
    expected = model.state_dict().keys()
    if weights.keys() != expected and all(name.startswith("module.") for name in weights):
        stripped = {name[7:]: value for name, value in weights.items()}
        if stripped.keys() == expected:
            weights = stripped
    model.load_state_dict(weights, strict=True)
    return checkpoint


class FrameDataset(Dataset):
    """Original VAE frame enumeration and bottom/right padding, without unused images."""
    def __init__(self, path, frame_st=0, frame_end=None):
        self.path = str(path)
        self.index = []
        self.file = None
        with h5py.File(self.path, "r") as f:
            for key in f:
                if key == "mask":
                    continue
                trials, frames, _, _ = f[key].shape
                end = frames if frame_end is None else min(frames, frame_end)
                self.index.extend((key, trial, t) for trial in range(trials) for t in range(frame_st, end))
        if not self.index:
            raise ValueError(f"No frames in {path}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.path, "r")
        key, trial, t = self.index[index]
        x = self.file[key][trial, t:t+1]
        h, w = x.shape[-2:]
        x = np.pad(x, ((0,0),(0,(-h)%16),(0,(-w)%16)))
        return (torch.from_numpy(x.astype(np.float32)),)

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


def balanced_loader(datasets, batch_size, steps, shuffle, seed, collate=None, workers=0):
    joined = ConcatDataset(datasets)
    offsets = [0] + joined.cumulative_sizes
    sampler = BalancedMultiSubjectBatchSampler([len(ds) for ds in datasets], offsets,
                                               batch_size, steps, shuffle, seed)
    return DataLoader(joined, batch_sampler=sampler, num_workers=workers, collate_fn=collate)


def train_vae(config, output, device="cpu"):
    seed_all(config.get("seed", 42))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model = UNet2D(**config["vae"]["model"]).to(device)
    settings = dict(w_pyr=5., w_sobel=1., w_msssim=0., w_temp=0., recon_loss_type="mse",
                    pyr_scales=3, channels_as_time=False, temp_mode="diff_l1", grad_clip=1.,
                    log_interval=100, epochs=50, batch_size=32, lr=1e-4, weight_decay=0.,
                    train_steps=-1, val_steps=-1, beta_start=0., beta_end=1e-3, warmup_steps=1000)
    settings.update(config["vae"].get("training", {}))
    args = SimpleNamespace(**settings)
    loaders = {}
    for split in ("train", "val"):
        datasets = [FrameDataset(s[split], **config.get("frame_range", {})) for s in config["subjects"]]
        loaders[split] = balanced_loader(datasets, args.batch_size,
            getattr(args, f"{split}_steps"), split=="train", config.get("seed",42)+(10000 if split=="val" else 0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    beta = BetaScheduler(args.warmup_steps, args.beta_start, args.beta_end)
    logger = logging.getLogger(__name__)
    best, step, records = float("inf"), 0, []
    for epoch in range(args.epochs):
        tr, step = vae_engine.train_one_epoch(model, loaders["train"], optimizer, device, epoch,
                                              args, beta, logger, step)
        va = vae_engine.evaluate(model, loaders["val"], device, args, beta.get_beta())
        record = dict(epoch=epoch, train=tr, val=va); records.append(record)
        checkpoint = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                          epoch=epoch, global_step=step, config=config)
        torch.save(checkpoint, output/"last.pt")
        if va["total_loss"] < best:
            best = va["total_loss"]; torch.save(checkpoint, output/"best.pt")
        print(json.dumps(record), flush=True)
    (output/"history.json").write_text(json.dumps(records, indent=2))
    for loader in loaders.values():
        for ds in loader.dataset.datasets: ds.close()
    return records


@torch.no_grad()
def compute_latent_stats(loader, vae, device, channels, max_batches=200):
    """Same accumulation order and variance arithmetic as the source trainer."""
    sums = torch.zeros(channels, dtype=torch.float64)
    squares = torch.zeros(channels, dtype=torch.float64)
    count = 0
    for i, batch in enumerate(loader):
        if max_batches != -1 and i >= max_batches: break
        hist, target = encode_frames_with_vae(vae, batch.history_clip.to(device), batch.target_clip.to(device))
        volume = torch.cat([hist,target], dim=2)
        sums += volume.sum(dim=(0,2,3,4)).double().cpu()
        squares += (volume*volume).sum(dim=(0,2,3,4)).double().cpu()
        count += volume.shape[0]*volume.shape[2]*volume.shape[3]*volume.shape[4]
    if not count: raise ValueError("Latent statistics require nonempty training batches")
    mean = (sums/count).float()
    variance = (squares/count-mean.double()*mean.double()).clamp(min=0).float()
    return LatentStats(mean, torch.sqrt(variance+1e-8).clamp(min=1e-6), count)


def build_datasets(config, split, store):
    window = config["diffusion"]["window"]
    datasets = [SlidingWindowLatentDataset(".", s[split], image_store=store, **window) for s in config["subjects"]]
    return MultiSubjectSlidingWindowDataset(datasets, [s["name"] for s in config["subjects"]])


def diffusion_loader(dataset, batch_size, steps=-1, shuffle=True, seed=42):
    sampler = BalancedMultiSubjectBatchSampler(dataset.subject_lengths, dataset.offsets,
                                               batch_size, steps, shuffle, seed)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)


def build_scheduler(config):
    return DDIMScheduler(num_train_timesteps=config.get("diffusion_steps",1000),
        beta_schedule=config.get("beta_schedule","squaredcos_cap_v2"),
        prediction_type="sample" if config.get("prediction_type","x0")=="x0" else "epsilon", clip_sample=False)


def build_models(config, store, device):
    vae = UNet2D(**config["vae"]["model"]).to(device)
    load_weights(vae, config["vae"]["checkpoint"], device=device)
    vae.eval().requires_grad_(False)
    settings = dict(config["diffusion"]["model"])
    settings.update(latent_channels=config["vae"]["model"]["latent_dim"], image_emb_dim=store.token_dim,
                    tokens_per_emb=store.tokens_per_emb, num_subjects=len(config["subjects"]))
    return vae, MultiSubjectSlidingWindowModel(**settings).to(device)


def train_diffusion(config, output, device="cpu"):
    device=torch.device(device)
    seed=config.get("seed",42); seed_all(seed)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    store=ImageEmbeddingStore(config["image_embeddings"],config.get("image_emb_dim",768))
    vae,model=build_models(config,store,device)
    defaults=dict(epochs=50,batch_size=8,lr=1e-4,weight_decay=1e-4,grad_clip=1.,
        history_noise_prob=0.,history_noise_p_mean=-3.,history_noise_p_std=2.,history_drop_prob=.1,
        cfg_drop_prob=.1,prob_focus_present=0.,prediction_type="x0",ema=False,ema_decay=.995,
        ema_update_every=10,ema_start_step=2000,eval_with_ema=True,train_steps=-1,val_steps=-1,latent_stats_max_batches=200)
    defaults.update(config["diffusion"].get("training",{})); args=SimpleNamespace(**defaults)
    train=build_datasets(config,"train",store); val=build_datasets(config,"val",store)
    train_loader=diffusion_loader(train,args.batch_size,args.train_steps,True,seed)
    val_loader=diffusion_loader(val,args.batch_size,args.val_steps,False,seed+10000)
    stats=compute_latent_stats(train_loader,vae,device,config["vae"]["model"]["latent_dim"],args.latent_stats_max_batches)
    torch.save(dict(mean=stats.mean,std=stats.std,count=stats.count),output/"latent_stats.pt")
    stats=stats.to(device)
    scheduler=build_scheduler(defaults)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    ema_model=copy.deepcopy(model).eval().requires_grad_(False) if args.ema else None
    ema=EMA(args.ema_decay) if args.ema else None
    state=TrainState(global_step=0); best=float("inf"); records=[]
    for epoch in range(args.epochs):
        tr=engine.train_one_epoch(model,ema,ema_model,scheduler,train_loader,optimizer,device,epoch,args,
            logging.getLogger(__name__),state,vae,stats,train,None,config["diffusion"]["window"]["target_frames"])
        eval_model=ema_model if args.ema and args.eval_with_ema else model
        va=engine.evaluate(eval_model,scheduler,val_loader,device,args,vae,stats,config["diffusion"]["window"]["target_frames"])
        checkpoint=dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
            ema_state_dict=ema_model.state_dict() if ema_model is not None else None,
            epoch=epoch,global_step=state.global_step,config=config,
            latent_stats=dict(mean=stats.mean.cpu(),std=stats.std.cpu(),count=stats.count))
        torch.save(checkpoint,output/"last.pt")
        if va<best: best=va;torch.save(checkpoint,output/"best.pt")
        record=dict(epoch=epoch,train_mse=tr,val_mse=va);records.append(record);print(json.dumps(record),flush=True)
    train.close();val.close()
    (output/"history.json").write_text(json.dumps(records,indent=2))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True);parser.add_argument("--stage",choices=["vae","diffusion"],required=True)
    parser.add_argument("--output",required=True);parser.add_argument("--device",default="cpu")
    a=parser.parse_args(); config=json.loads(Path(a.config).read_text())
    (train_vae if a.stage=="vae" else train_diffusion)(config,a.output,a.device)

if __name__=="__main__": main()
