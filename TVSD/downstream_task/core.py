"""Original conditional image-diffusion objective and sampling operations."""
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio


def train_one_step(model, vae, optimizer, scheduler, images, responses,
                   condition_drop=0.1, subject_slice=None, device='cpu'):
    model.train()
    optimizer.zero_grad()
    images = images.float().to(device)
    with torch.no_grad():
        latents, _ = vae.encode(images)
        if subject_slice is not None:
            responses = responses[:, subject_slice]
        condition = responses.unsqueeze(1).to(device)
        if condition_drop > 0:
            dropped = torch.zeros(latents.shape[0], device=latents.device).float().uniform_(0, 1) < condition_drop
            condition[dropped, :, :] = torch.zeros((1, condition.shape[-1]), device=device).unsqueeze(0)
    noise = torch.randn_like(latents).to(device)
    t = torch.randint(0, scheduler.num_timesteps, (latents.shape[0],)).to(device)
    noisy = scheduler.add_noise(latents, noise, t)
    prediction = model(noisy, t, cond_input=condition)
    loss = torch.nn.functional.mse_loss(prediction, noise)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def validation_loss(model, vae, loader, scheduler, subject_slice=None, device='cpu'):
    model.eval()
    losses = []
    for images, responses in loader:
        latents, _ = vae.encode(images.float().to(device))
        if subject_slice is not None:
            responses = responses[:, subject_slice]
        condition = responses.unsqueeze(1).to(device)
        noise = torch.randn_like(latents).to(device)
        t = torch.randint(0, scheduler.num_timesteps, (latents.shape[0],)).to(device)
        prediction = model(scheduler.add_noise(latents, noise, t), t, cond_input=condition)
        losses.append(torch.nn.functional.mse_loss(prediction, noise).item())
    if not losses:
        raise ValueError('Validation dataset is empty')
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def reconstruct(model, vae, scheduler, responses, image_size, autoencoder_config, device='cpu'):
    """Full ancestral DDPM sampling; changing the number of steps changes the model schedule."""
    model.eval()
    vae.eval()
    latent_size = image_size // 2**sum(autoencoder_config['down_sample'])
    x = torch.randn((len(responses), autoencoder_config['z_channels'], latent_size, latent_size), device=device)
    condition = responses.unsqueeze(1).to(device)
    for step in reversed(range(scheduler.num_timesteps)):
        t = torch.full((len(responses),), step, dtype=torch.long, device=device)
        prediction = model(x, t, cond_input=condition)
        x, _ = scheduler.sample_prev_timestep(x, prediction, torch.as_tensor(step, device=device))
    return (vae.decode(x).clamp(-1, 1).detach().cpu()+1)/2


@torch.no_grad()
def compute_pair_metrics(pred_tensor_01, true_tensor_01, lpips_fn):
    """Reference PSNR quantizes to uint8; LPIPS uses continuous [-1,1] tensors."""
    pred = pred_tensor_01.detach().cpu().clamp(0, 1)
    true = true_tensor_01.detach().cpu().clamp(0, 1)
    pred_u8 = (pred.permute(1, 2, 0).numpy()*255).astype(np.uint8)
    true_u8 = (true.permute(1, 2, 0).numpy()*255).astype(np.uint8)
    psnr = peak_signal_noise_ratio(true_u8, pred_u8, data_range=255)
    device = next(lpips_fn.parameters()).device
    lpips = lpips_fn(2*(pred.unsqueeze(0).to(device)-0.5),
                     2*(true.unsqueeze(0).to(device)-0.5)).item()
    return float(psnr), float(lpips)
