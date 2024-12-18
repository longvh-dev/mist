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

def preprocess_image(image):
    # image = Image.open(path).convert("RGB")
    w, h = image.size
    # print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
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
    seed: int = 42
    ckpt: str = "models/ldm/stable-diffusion-v1-4/model.ckpt"
    config: str = "configs/stable-diffusion/v1-inference-attack.yaml"
    prompt: str = "a portrait"
    outdir: str = "outputs"
    from_file: str = None
    init_img: str = "test/sample.png"
    strength: float = 0.3
    plms: bool = False
    skip_save: bool = False
    skip_grid: bool = False
    # watermark: str = "watermark.png"

def run_diffusion(model, init_image, prompt, strength=0.3, **kwargs):
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

    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    print(f"init image shape: {init_image.shape}")
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                for n in trange(opt.n_iter, desc="Sampling"):
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    c = model.get_learned_conditioning(batch_size * [prompt])

                    # encode (scaled latent)
                    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                    print(f"Mismatch in batch sizes: z_enc={z_enc.shape[0]}, c={c.shape[0]}")
                    print(f"z_enc shape: {z_enc.shape}, c shape: {c.shape}")
                    print(f"uc shape: {uc.shape if uc is not None else None}")
                    
                    assert uc is None or uc.shape[0] == c.shape[0], "Unconditional conditioning must match batch size"

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


# def run_diffusion(model, prompt, init_image, strength=0.3, ddim_steps=50, **kwargs):
#     opt = Config()
#     for k, v in kwargs.items():
#         setattr(opt, k, v)
        
#     print(f"Running diffusion with config: {opt}")
#     seed_everything(opt.seed)
#     device = next(model.parameters()).device

#     sampler = DDIMSampler(model)

#     # os.makedirs(opt.outdir, exist_ok=True)
#     # outpath = opt.outdir

#     batch_size = opt.n_samples
#     n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
#     if not opt.from_file:
#         prompt = prompt
#         assert prompt is not None
#         data = [batch_size * [prompt]]

#     # else:
#     #     print(f"reading prompts from {opt.from_file}")
#     #     with open(opt.from_file, "r") as f:
#     #         data = f.read().splitlines()
#     #         data = list(chunk(data, batch_size))

#     sample_path = os.path.join(outpath, "samples")
#     os.makedirs(sample_path, exist_ok=True)
#     base_count = len(os.listdir(sample_path))
#     grid_count = len(os.listdir(outpath)) - 1

#     # assert os.path.isfile(opt.init_img)
#     # init_image = load_img(opt.init_img).to(device)
#     init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
#     init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

#     sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

#     assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
#     t_enc = int(opt.strength * opt.ddim_steps)
#     print(f"target t_enc is {t_enc} steps")

#     precision_scope = autocast if opt.precision == "autocast" else nullcontext
#     with torch.no_grad():
#         with precision_scope("cuda"):
#             with model.ema_scope():
#                 tic = time.time()
#                 all_samples = list()
#                 for n in trange(opt.n_iter, desc="Sampling"):
#                     for prompts in tqdm(data, desc="data"):
#                         uc = None
#                         if opt.scale != 1.0:
#                             uc = model.get_learned_conditioning(batch_size * [""])
#                         if isinstance(prompts, tuple):
#                             prompts = list(prompts)
#                         c = model.get_learned_conditioning(prompts)

#                         # encode (scaled latent)
#                         z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
#                         # decode it
#                         samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
#                                                  unconditional_conditioning=uc,)

#                         x_samples = model.decode_first_stage(samples)
#                         x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

#                         if not opt.skip_save:
#                             for x_sample in x_samples:
#                                 x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
#                                 Image.fromarray(x_sample.astype(np.uint8)).save(
#                                     os.path.join(sample_path, f"{base_count:05}.png"))
#                                 base_count += 1
#                         all_samples.append(x_samples)

#                 if not opt.skip_grid:
#                     # additionally, save as grid
#                     grid = torch.stack(all_samples, 0)
#                     grid = rearrange(grid, 'n b c h w -> (n b) c h w')
#                     grid = make_grid(grid, nrow=n_rows)

#                     # to image
#                     grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
#                     Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
#                     grid_count += 1

#                 toc = time.time()

#     print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
#           f" \nEnjoy.")
# def load_model_from_config(config, ckpt, verbose=False):
#     print(f"Loading model from {ckpt}")
#     pl_sd = torch.load(ckpt, map_location="cpu")
#     if "global_step" in pl_sd:
#         print(f"Global Step: {pl_sd['global_step']}")
#     sd = pl_sd["state_dict"]
#     model = instantiate_from_config(config.model)
#     m, u = model.load_state_dict(sd, strict=False)
#     if len(m) > 0 and verbose:
#         print("missing keys:")
#         print(m)
#     if len(u) > 0 and verbose:
#         print("unexpected keys:")
#         print(u)

#     model.cuda()
#     model.eval()
#     return model


# def load_img(path):
#     image = Image.open(path).convert("RGB")
#     w, h = image.size
#     print(f"loaded input image of size ({w}, {h}) from {path}")
#     w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32
#     # resize to square
#     image = image.resize((w, h), resample=PIL.Image.LANCZOS)
#     image = np.array(image).astype(np.float32) / 255.0
#     image = image[None].transpose(0, 3, 1, 2)
#     image = torch.from_numpy(image)
#     print(f"image shape: {image.shape}")
    # return 2.*image - 1.
# if __name__ == "__main__":
#     main()
