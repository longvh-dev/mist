import torch
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from models import Generator, Discriminator
from data_loader import create_dataloader
# from config import ModelConfig, TrainingConfig

def calculate_mse(img1, img2):
    """Calculate MSE between two images"""
    return F.mse_loss(img1, img2).item()

def preprocess_for_inception(images):
    """Convert float tensors (0-1) to uint8 format (0-255) for inception model"""
    # Denormalize if needed (assuming inputs are normalized to [-1,1] or [0,1])
    if images.min() < 0:
        images = (images + 1) / 2  # [-1,1] -> [0,1]
    
    # Convert to uint8
    images = (images * 255).clamp(0, 255).to(torch.uint8)
    
    # Ensure correct channel order (RGB)
    if images.shape[1] == 3:  # If channels first
        return images
    elif images.shape[-1] == 3:  # If channels last
        return images.permute(0, 3, 1, 2)
    else:
        raise ValueError("Input images must have 3 channels")

def evaluate_adversarial_quality(generator, dataloader, device):
    """Evaluate adversarial example quality metrics"""
    generator.eval()
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    
    mse_scores = []
    psnr_scores = []
    ssim_scores = []
    
    with torch.no_grad():
        for real_images, watermarks, _ in tqdm(dataloader, desc="Evaluating adversarial quality", disable=True):
            real_images = real_images.to(device)
            watermarks = watermarks.to(device)
            
            # Generate adversarial examples
            perturbation = generator(real_images, watermarks)
            adv_images = real_images + perturbation
            
            # Calculate metrics
            mse_scores.append(calculate_mse(real_images, adv_images))
            psnr_scores.append(psnr(adv_images, real_images).item())
            ssim_scores.append(ssim(adv_images, real_images).item())
    
    return {
        'mse': np.mean(mse_scores),
        'psnr': np.mean(psnr_scores),
        'ssim': np.mean(ssim_scores)
    }

def evaluate_generation_quality(real_images, generated_images, device):
    """Evaluate generated image quality metrics"""
    fid = FrechetInceptionDistance(normalize=True).to(device)
    
    # Preprocess images for inception model
    real_images = preprocess_for_inception(real_images)
    generated_images = preprocess_for_inception(generated_images)
    
    # Add images to FID
    for real, gen in zip(real_images, generated_images):
        fid.update(real.unsqueeze(0), real=True)
        fid.update(gen.unsqueeze(0), real=False)
    
    fid_score = fid.compute().item()
    
    # Calculate precision (using inception features)
    inception = InceptionScore().to(device)
    real_features = inception(real_images)
    gen_features = inception(generated_images)
    del inception
    torch.cuda.empty_cache()
    
    # Calculate precision as % of generated images close to real ones
    distances = torch.cdist(gen_features, real_features)
    precision = (distances.min(dim=1)[0] < 0.1).float().mean().item()
    
    return {
        'fid': fid_score,
        'precision': precision
    }

# def main():
#     # Load configs
#     # model_config = ModelConfig()
#     # train_config = TrainingConfig()
    
#     # Setup device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Load dataset
#     dataloader, _ = create_dataloader(
#         image_dir="data/wikiart",
#         classes_csv="data/wikiart/eval_classes.csv",
#         batch_size=8,
#         shuffle=False
#     )
    
#     # Load model
#     generator = Generator(model_config.input_channels).to(device)
#     checkpoint = torch.load('checkpoints/20241112-221929/checkpoint_epoch_200.pth')
#     generator.load_state_dict(checkpoint['generator_state_dict'])
#     generator.eval()
    
#     # Evaluate adversarial quality
#     adv_metrics = evaluate_adversarial_quality(generator, dataloader, device)
#     print("\nAdversarial Example Quality Metrics:")
#     print(f"MSE: {adv_metrics['mse']:.4f}")
#     print(f"PSNR: {adv_metrics['psnr']:.4f} dB")
#     print(f"SSIM: {adv_metrics['ssim']:.4f}")
#     torch.cuda.empty_cache()
    
#     # Collect real and generated images for FID
#     real_images = []
#     generated_images = []
    
#     with torch.no_grad():
#         for real_imgs, watermarks, _ in tqdm(dataloader, desc="Generating images"):
#             real_imgs = real_imgs.to(device)
#             watermarks = watermarks.to(device)
            
#             # Generate adversarial examples and diffusion outputs
#             perturbation = generator(real_imgs, watermarks)
#             adv_images = real_imgs + perturbation
            
#             real_images.append(real_imgs)
#             generated_images.append(adv_images)
            
#             # Clear cache to reduce memory usage
#             del real_imgs, watermarks, perturbation, adv_images
#             torch.cuda.empty_cache()
#     del generator
#     torch.cuda.empty_cache()
    
#     real_images = torch.cat(real_images)
#     generated_images = torch.cat(generated_images)
    
#     # Evaluate generation quality
#     gen_metrics = evaluate_generation_quality(real_images, generated_images, device)
#     print("\nGenerated Image Quality Metrics:")
#     print(f"FID: {gen_metrics['fid']:.4f}")
#     print(f"Precision: {gen_metrics['precision']:.4f}")

# if __name__ == "__main__":
#     main()