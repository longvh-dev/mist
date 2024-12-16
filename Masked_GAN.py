from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Generator, Discriminator, target_model
from evaluate import evaluate_adversarial_quality
from mist_utils import load_model_from_config

models_path = 'models/gan/'

def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif hasattr(m, 'weight') and classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

class GANAttack:
    """
    """
    def __init__(self, target_model, image_nc, device, box_min, box_max, config):
        """
        """

        self.target_model = target_model
        self.image_nc = image_nc
        self.device = device
        self.box_min = box_min
        self.box_max = box_max
        self.config = config
        
        self.netG = Generator(image_nc*2).to(self.device)
        self.netD = Discriminator(image_nc).to(self.device)
        
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.config.lr)
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=self.config.lr)
        
        self.model_path = os.path.join(models_path, time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()))
        os.makedirs(self.model_path)
        
    def train_step(self, real_images, watermark):
        # optimizer D
        for _ in range(1):
            perturbation = self.netG(real_images, watermark)
            # adv_images = torch.clamp(perturbation, -0.3, 0.3) + real_images
            adv_images = perturbation + real_images
            adv_images = torch.clamp(adv_images, 0, 1)
            
            self.optimizer_D.zero_grad()
            
            pred_real = self.netD(real_images)
            loss_D_real = F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()
            
            pred_fake = self.netD(adv_images.detach())
            loss_D_fake = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward()
            
            loss_D_GAN = loss_D_real + loss_D_fake
            self.optimizer_D.step()

        # Train Generator
        for _ in range(1):
            self.optimizer_G.zero_grad()
            
            pred_fake = self.netD(adv_images)
            loss_G_fake = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # Adversarial Loss
            z_adv_images, _ = self.target_model.get_components(adv_images, True)
            z_watermark, _ = self.target_model.get_components(watermark, True)
            loss_adv = torch.norm(z_adv_images - z_watermark, p=2, dim=1).mean()
            
            # Perturbation Loss
            weighted_perturbation = perturbation * (1 + watermark * self.config.watermark_region)
            weighted_perturbation = torch.norm(weighted_perturbation, p=2, dim=1)
            mask = ((weighted_perturbation - self.config.c)>0) + 0 
            loss_pert = (weighted_perturbation * mask).mean()
            
            # sum loss
            g_loss = loss_adv  + self.config.beta * loss_pert

            g_loss.backward()
            self.optimizer_G.step()
            
            g_loss_sum = config.alpha * loss_G_fake + loss_adv + self.config.beta * loss_pert

        return loss_D_GAN.item(), loss_G_fake.item(), loss_adv.item(), loss_pert.item(), g_loss_sum.item()
    
    def train(self, train_dataloaders, eval_dataloaders):
        """
        """
        for epoch in range(self.config.epochs):
            self.netG.train()
            self.netD.train()
            
            if epoch == 40:
                self.optimizerG.param_groups[0]['lr'] /= 10
                self.optimizerD.param_groups[0]['lr'] /= 10
            
            for i, data in enumerate(train_dataloaders):
                real_images, watermark, _ = data
                real_images = real_images.to(self.device)
                watermark = watermark.to(self.device)
                
                loss_D_GAN, loss_G_fake, loss_adv, loss_pert, g_loss_sum = self.train_step(real_images, watermark)
                
                print(f"Epoch [{epoch}/{args.num_epochs}]\tBatch [{batch_idx}]\t"
                      f"loss_D_GAN: {loss_D_GAN:.8f}\t"
                      f"loss_G_fake: {loss_G_fake:.8f}\t"
                      f"loss_adv: {loss_adv:.8f}\t"
                      f"loss_pert: {loss_pert:.8f}\t"
                      f"loss_G_sum: {loss_G_sum:.8f}\t")
                
            
            if epoch % 5 == 0 and epoch != 0:
                self.netG.eval()

                adv_metrics = evaluate_adversarial_quality(self.netG, eval_dataloaders, device)

                print("\nAdversarial Example Quality Metrics:")
                print(f"MSE: {adv_metrics['mse']:.8f}")
                print(f"PSNR: {adv_metrics['psnr']:.8f} dB")
                print(f"SSIM: {adv_metrics['ssim']:.8f}")
            
            if epoch % 10 == 0 and epoch != 0:
                torch.save(self.netG.state_dict(), os.path.join(self.model_path, f'netG_{epoch}.pth'))
                # torch.save(self.netD.state_dict(), os.path.join(self.model_path, f'netD_{epoch}.pth'))
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GAN Attack')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha')
    parser.add_argument('--beta', type=float, default=10.0, help='Beta')
    parser.add_argument('--c', type=float, default=10/255, help='c')
    parser.add_argument('--watermark_region', type=float, default=4.0, help='Watermark region')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ### LDM
    if ckpt is None:
        ckpt = 'models/ldm/stable-diffusion-v1-4/model.ckpt'

    if base is None:
        base = 'configs/stable-diffusion/v1-inference-attack.yaml'

    seed_everything(seed)
    imagenet_templates_small_style = ['a painting']
    imagenet_templates_small_object = ['a photo']

    config_path = os.path.join(os.getcwd(), base)
    config = OmegaConf.load(config_path)

    ckpt_path = os.path.join(os.getcwd(), ckpt)
    model = load_model_from_config(config, ckpt_path).to(device)

    fn = nn.MSELoss(reduction="sum")

    input_prompt = [imagenet_templates_small_object[0] for i in range(1)]
    net = target_model(model, input_prompt, mode=mode, rate=rate, device=device)
    net.eval()
    ### 
    
    train_dataloaders = None
    eval_dataloaders = None
    
    gan_attack = GANAttack(target_model, 3, device, 0, 1, args)
    gan_attack.train(train_dataloaders, eval_dataloaders)
    
    torch.save(gan_attack.netG.state_dict(), os.path.join(gan_attack.model_path, 'netG.pth'))
    torch.save(gan_attack.netD.state_dict(), os.path.join(gan_attack.model_path, 'netD.pth'))