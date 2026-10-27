"""Train/evaluate a 78-class decoder on real, synthetic, mixed, or finetuned data."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, ConcatDataset
from .model import build_decoder
from .metrics import CrossEntropyLoss, Accuracy
from .data import ResponseDataset
from ..diffusion.train import seed_all, load_weights


def optimizer_for(model,lr=1e-4):
    # Preserve the actual source Trainer: Adam, not AdamW.
    return torch.optim.Adam(model.parameters(),lr=lr,weight_decay=.05)


def step(model,optimizer,inputs,targets,device="cpu",update=True):
    inputs=[x.to(device) for x in inputs];targets=[y.to(device) for y in targets]
    if update:optimizer.zero_grad()
    logits=model(inputs)
    loss=CrossEntropyLoss()(logits,targets)
    if update:loss.backward();optimizer.step()
    accuracy=Accuracy()([logits[0].detach()],targets)
    return float(loss.detach()),float(accuracy),logits


def iterate(model,loader,optimizer=None,device="cpu",max_steps=None):
    update=optimizer is not None;model.train(update)
    total=loss_sum=acc_sum=0
    with torch.set_grad_enabled(update):
        for i,(inputs,targets) in enumerate(loader):
            loss,acc,_=step(model,optimizer,inputs,targets,device,update)
            n=inputs[0].shape[0];total+=n;loss_sum+=loss*n;acc_sum+=acc*n
            if max_steps is not None and i+1>=max_steps:break
    if not total:raise ValueError("Cannot evaluate an empty loader")
    return dict(cross_entropy=loss_sum/total,accuracy=acc_sum/total,n=total)


def fit(model,train_loader,val_loader,output,epochs=100,lr=1e-4,device="cpu",
        patience=5,lr_decay_after=10,lr_decay_factor=10.,config=None,initial_checkpoint=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    optimizer=optimizer_for(model,lr);best=float("inf");bad=0;records=[];best_state=None
    if initial_checkpoint is not None:
        checkpoint=load_weights(model,initial_checkpoint,device=device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for epoch in range(epochs):
        tr=iterate(model,train_loader,optimizer,device);va=iterate(model,val_loader,device=device)
        row=dict(epoch=epoch,train=tr,val=va);records.append(row)
        checkpoint=dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                        epoch=epoch,config=config)
        torch.save(checkpoint,output/"last.pt")
        if va["cross_entropy"]<best:
            best=va["cross_entropy"];bad=0;best_state=copy.deepcopy(model.state_dict())
            torch.save(checkpoint,output/"best.pt")
        else:
            bad+=1
            if bad>=patience:break
            if bad%lr_decay_after==0:
                for group in optimizer.param_groups:group["lr"]/=lr_decay_factor
        print(json.dumps(row),flush=True)
    if best_state is None:raise RuntimeError("No finite validation checkpoint was selected")
    model.load_state_dict(best_state)
    (output/"history.json").write_text(json.dumps(records,indent=2))
    return records


def run(config,output,mode="real",device="cpu",evaluate_only=None):
    seed_all(config.get("seed",0));output=Path(output);output.mkdir(parents=True,exist_ok=True)
    batch=config.get("batch_size",64)
    real_window=config.get("real_window",[20,28]);syn_window=config.get("synthetic_window",[0,8])
    val=ResponseDataset(config["val"],*real_window);test=ResponseDataset(config["test"],*real_window)
    datasets=[val,test]
    val_loader=DataLoader(val,batch_size=batch,shuffle=True)
    test_loader=DataLoader(test,batch_size=batch,shuffle=False)
    shape=tuple(val[0][0][0].shape[1:]);val.close()
    nodes=config.get("nodes")
    if nodes is None:nodes=json.loads(Path(__file__).with_name("architecture.json").read_text())
    model=build_decoder(nodes,shape,config.get("num_classes",78)).to(device)
    if evaluate_only:
        load_weights(model,evaluate_only,device=device)
    else:
        phases=[]
        if mode in {"real","mixed","finetune"}:
            real=ResponseDataset(config["real_train"],*real_window);datasets.append(real)
        if mode in {"synthetic","mixed","finetune"}:
            synthetic=ResponseDataset(config["synthetic_train"],*syn_window);datasets.append(synthetic)
        if mode=="real":phases=[("real",real)]
        elif mode=="synthetic":phases=[("synthetic",synthetic)]
        elif mode=="mixed":phases=[("mixed",ConcatDataset([real,synthetic]))]
        elif mode=="finetune":phases=[("synthetic",synthetic),("finetune",real)]
        else:raise ValueError(mode)
        for phase,dataset in phases:
            train_loader=DataLoader(dataset,batch_size=batch,shuffle=True,drop_last=True)
            fit(model,train_loader,val_loader,output/phase,epochs=config.get("epochs",100),
                lr=config.get("lr",1e-4),device=device,patience=config.get("patience",5),config=config,
                initial_checkpoint=output/"synthetic"/"best.pt" if phase=="finetune" else None)
            metrics=iterate(model,test_loader,device=device)
            (output/f"{phase}_test.json").write_text(json.dumps(metrics,indent=2))
    metrics=iterate(model,test_loader,device=device)
    (output/"test_metrics.json").write_text(json.dumps(metrics,indent=2))
    for ds in datasets:ds.close()
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument("--mode",choices=["real","synthetic","mixed","finetune"],default="real")
    p.add_argument("--device",default="cpu");p.add_argument("--evaluate-checkpoint")
    a=p.parse_args();print(json.dumps(run(json.loads(Path(a.config).read_text()),a.output,a.mode,a.device,a.evaluate_checkpoint)))

if __name__=="__main__":main()
