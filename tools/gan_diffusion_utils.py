import argparse, os, sys, glob
import PIL
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from contextlib import nullcontext
import time
from dataclasses import dataclass

from einops import rearrange, repeat
import torch
from torch import autocast
from torchvision.utils import make_grid
from torchvision import transforms
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

# def preprocess_image(image, transform, device):
#     image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
#     return image

def generate_adversarial_image(netG, image, watermark, box_min, box_max):
    perturbation = netG(image, watermark)
    adv_image = perturbation + image
    adv_image_clamp = torch.clamp(perturbation, -0.3, 0.3) + image
    adv_image_clamp = torch.clamp(adv_image_clamp, box_min, box_max)
    return adv_image, adv_image_clamp

def save_tensor_image(tensor, path):
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = (tensor + 1) / 2  # De-normalize
    tensor = tensor.clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    image.save(path)

def preprocess_image(image, device):
    # image = Image.open(path).convert("RGB")
    w, h = image.size
    # print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(device)
    return 2.*image - 1.


@dataclass
class Config:
    ddim_steps: int = 50
    ddim_eta: float = 0.0
    n_iter: int = 1
    C: int = 4
    f: int = 8
    n_samples: int = 1
    n_rows: int = 0
    scale: float = 5.0
    precision: str = "autocast"

def run_diffusion(model, prompt, init_image, strength=0.3, ddim_steps=50, **kwargs):
    device = next(model.parameters()).device
    opt = Config()
    # update config with kwargs
    for k, v in kwargs.items():
        setattr(opt, k, v)
    
    batch_size = opt.n_samples
    

    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                for n in trange(opt.n_iter, desc="Sampling"):
                    uc = None
                    c = model.get_learned_conditioning(prompt)

                    # encode (scaled latent)
                    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                    # decode it
                    samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,)

                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    print(f"sampled image of size {x_samples.shape}")

                    return x_samples
                    # all_samples.append(x_samples)

                toc = time.time()

    print(f"Sampling took {toc - tic:.2f} seconds")
    print("----------------------------------------")


if __name__ == "__main__":
    main()
