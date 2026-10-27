"""Explicit real/synthetic H5 adapter with the source decoder's identity normalization."""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ResponseDataset(Dataset):
    def __init__(self,path,frame_start=20,frame_end=28):
        self.path=str(path);self.start=frame_start;self.end=frame_end;self.file=None;self.index=[]
        with h5py.File(self.path,"r") as f:
            for key in f:
                if key=="mask":continue
                if not key.startswith("image_ids"):raise ValueError(f"Expected image_ids<int>, got {key}")
                node=f[key]
                if isinstance(node,h5py.Group):
                    if "samples" not in node:raise ValueError(f"Synthetic group {key} needs samples")
                    node=node["samples"]
                if self.start<0 or self.end>node.shape[1] or self.end<=self.start:
                    raise ValueError(f"Invalid frame window [{self.start},{self.end}) for {key}: {node.shape}")
                self.index.extend((key,i) for i in range(len(node)))
        if not self.index:raise ValueError(f"No response trials in {path}")

    def __len__(self):return len(self.index)

    def __getitem__(self,index):
        if self.file is None:self.file=h5py.File(self.path,"r")
        key,trial=self.index[index];node=self.file[key]
        if isinstance(node,h5py.Group):node=node["samples"]
        # Source loader selects one trial, crops to at most 360 pixels, leaves identity normalization.
        x=node[trial:trial+1,self.start:self.end,:360,:360].mean(0)
        return [torch.from_numpy(x[None].astype(np.float32))],[int(key.removeprefix("image_ids"))]

    def close(self):
        if self.file is not None:self.file.close();self.file=None
