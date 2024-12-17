import torch
from torchvision import transforms
from PIL import Image

def preprocess_image(image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    return image

def generate_adversarial_image(netG, image, watermark, box_min, box_max):
    perturbation = netG(image, watermark)
    adv_image = perturbation + image
    adv_image_clamp = torch.clamp(perturbation, -0.3, 0.3) + image
    adv_image_clamp = torch.clamp(adv_image_clamp, box_min, box_max)
    return adv_image, adv_image_clamp

def run_diffusion_model(diffusion_model, adv_image, strength):
    timesteps = int(diffusion_model.num_timesteps * strength)
    with torch.no_grad():
        output_image, _ = diffusion_model.sample(cond=None, batch_size=1, timesteps=timesteps, x_T=adv_image)
    return output_image

def save_tensor_image(tensor, path):
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = (tensor + 1) / 2  # De-normalize
    tensor = tensor.clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    image.save(path)