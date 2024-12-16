import os
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
from einops import rearrange
import ssl
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything

from Masked_PGD import LinfPGDAttack
from models import target_model
from mist_utils import parse_args, load_mask, closing_resize, load_image_from_path, load_model_from_config


ssl._create_default_https_context = ssl._create_unverified_context
os.environ['TORCH_HOME'] = os.getcwd()
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hub/')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class identity_loss(nn.Module):
    """
    An identity loss used for input fn for advertorch. To support semantic loss,
    the computation of the loss is implemented in class targe_model.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x


def init(epsilon: int = 16, steps: int = 100, alpha: int = 1,
         input_size: int = 512, object: bool = False, seed: int = 23,
         ckpt: str = None, base: str = None, mode: int = 2, rate: int = 10000):
    """
    Prepare the config and the model used for generating adversarial examples.
    :param epsilon: Strength of adversarial attack in l_{\infinity}.
                    After the round and the clip process during adversarial attack,
                    the final perturbation budget will be (epsilon+1)/255.
    :param steps: Iterations of the attack.
    :param alpha: strength of the attack for each step. Measured in l_{\infinity}.
    :param input_size: Size of the input image.
    :param object: Set True if the targeted images describes a specifc object instead of a style.
    :param mode: The mode for computation of the loss. 0: semantic; 1: textural; 2: fused.
                 See the document for more details about the mode.
    :param rate: The fusion weight. Higher rate refers to more emphasis on semantic loss.
    :returns: a dictionary containing model and config.
    """

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

    fn = identity_loss()

    if object:
        imagenet_templates_small = imagenet_templates_small_object
    else:
        imagenet_templates_small = imagenet_templates_small_style

    input_prompt = [imagenet_templates_small[0] for i in range(1)]
    net = target_model(model, input_prompt, mode=mode, rate=rate, device=device)
    net.eval()

    # parameter
    parameters = {
        'epsilon': epsilon/255.0 * (1-(-1)),
        'alpha': alpha/255.0 * (1-(-1)),
        'steps': steps,
        'input_size': input_size,
        'mode': mode,
        'rate': rate
    }

    return {'net': net, 'fn': fn, 'parameters': parameters}


def infer(img: PIL.Image.Image, config, tar_img: PIL.Image.Image = None, mask: PIL.Image.Image = None) -> np.ndarray:
    """
    Process the input image and generate the misted image.
    :param img: The input image or the image block to be misted.
    :param config: config for the attack.
    :param img: The target image or the target block as the reference for the textural loss.
    :returns: A misted image.
    """

    net = config['net']
    fn = config['fn']
    parameters = config['parameters']
    mode = parameters['mode']
    epsilon = parameters["epsilon"]
    alpha = parameters["alpha"]
    steps = parameters["steps"]
    input_size = parameters["input_size"]
    rate = parameters["rate"]
    trans = transforms.Compose([transforms.ToTensor()])

    img = np.array(img).astype(np.float32) / 127.5 - 1.0
    img = img[:, :, :3]
    if tar_img is not None:
        tar_img = np.array(tar_img).astype(np.float32) / 127.5 - 1.0
        tar_img = tar_img[:, :, :3]
    if mask is not None:
        mask = load_mask(mask).astype(np.float32) / 255.0
        mask = mask[:, :, :3]
        mask = trans(mask).unsqueeze(0).to(device)

    # data_source = torch.zeros([1, 3, input_size, input_size]).to(device)
    data_source = torch.zeros([1, 3, img.shape[0], img.shape[1]]).to(device)
    data_source[0] = trans(img).to(device)

    # target_info = torch.zeros([1, 3, input_size, input_size]).to(device)
    target_info = torch.zeros([1, 3, img.shape[0], img.shape[1]]).to(device)
    target_info[0] = trans(tar_img).to(device)
    # net.target_info = target_info
    net.target_size = input_size
    net.mode = mode
    net.rate = rate
    label = torch.zeros(data_source.shape).to(device)
    print(net(data_source, target_info, components=True))

    # Targeted PGD attack is applied.
    attack = LinfPGDAttack(net, fn, epsilon, steps, eps_iter=alpha, clip_min=-1.0, targeted=True)
    attack_output = attack.perturb(data_source, label, mask=mask)
    print(net(attack_output, target_info, components=True))

    output = attack_output[0]
    save_adv = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0).detach()
    grid_adv = 255. * rearrange(save_adv, 'c h w -> h w c').cpu().numpy()
    grid_adv = grid_adv
    return grid_adv


# Test the script with command: python mist_v3.py -img test/sample.png --output_name misted_sample
# For low Vram cost, test the script with command: python mist_v3.py -img test/sample.png --output_name misted_sample --block_num 2
# Test the new functions:  python mist_v3.py -img test/sample_random_size.png --output_name misted_sample --mask --non_resize --mask_path test/processed_mask.png

# Test the script for Vangogh dataset with command: python mist_v3.py -inp test/vangogh --output_dir vangogh
# For low Vram cost, test the script with command: python mist_v3.py -inp test/vangogh --output_dir vangogh --block_num 2

if __name__ == "__main__":
    args = parse_args()
    epsilon = args.epsilon
    steps = args.steps
    input_size = args.input_size
    block_num = args.block_num
    mode = args.mode
    rate = 10 ** (args.rate + 3)
    mask = args.mask
    resize = args.non_resize
    print(epsilon, steps, input_size, block_num, mode, rate, mask, resize)
    target_image_path = 'MIST.png'
    bls = input_size//block_num
    if args.input_dir_path:
        image_dir_path = args.input_dir_path
        config = init(epsilon=epsilon, steps=steps, mode=mode, rate=rate)
        config['parameters']["input_size"] = bls

        for img_id in os.listdir(image_dir_path):
            image_path = os.path.join(image_dir_path, img_id)

            if resize:
                img, target_size = closing_resize(image_path, input_size, block_num)
                bls_h = target_size[0]//block_num
                bls_w = target_size[1]//block_num
                tar_img = load_image_from_path(target_image_path, target_size[0],
                                               target_size[1])
            else:
                img = load_image_from_path(image_path, input_size)
                tar_img = load_image_from_path(target_image_path, input_size)
                bls_h = bls_w = bls
                target_size = [input_size, input_size]
            output_image = np.zeros([target_size[1], target_size[0], 3])
            if mask:
                print("Alert: Mask function is disabled when processed images in dir. Please set input_dir_path as None to enable mask.")
            processed_mask = None
            for i in tqdm(range(block_num)):
                for j in tqdm(range(block_num)):
                    if processed_mask is not None:
                        input_mask = Image.fromarray(np.array(processed_mask)[bls_w*i: bls_w*i+bls_w, bls_h*j: bls_h*j + bls_h])
                    else:
                        input_mask = None
                    img_block = Image.fromarray(np.array(img)[bls_w*i: bls_w*i+bls_w, bls_h*j: bls_h*j + bls_h])
                    tar_block = Image.fromarray(np.array(tar_img)[bls_w*i: bls_w*i+bls_w, bls_h*j: bls_h*j + bls_h])

                    output_image[bls_w*i: bls_w*i+bls_w, bls_h*j: bls_h*j + bls_h] = infer(img_block, config, tar_block, input_mask)
            output = Image.fromarray(output_image.astype(np.uint8))
            output_dir = os.path.join('outputs/dirs', args.output_dir)
            class_name = '_' + str(epsilon) + '_' + str(steps) + '_' + str(input_size) + '_' + str(block_num) + '_' + str(mode) + '_' + str(args.rate) + '_' + str(int(mask)) + '_' + str(int(resize))
            output_path_dir = output_dir + class_name
            if not os.path.exists(output_path_dir):
                os.mkdir(output_path_dir)
            output_path = os.path.join(output_path_dir, img_id)
            print("Output image saved in path {}".format(output_path))
            output.save(output_path)
    else:
        image_path = args.input_image_path
        if resize:
            img, target_size = closing_resize(image_path, input_size, block_num)
            bls_h = target_size[0]//block_num
            bls_w = target_size[1]//block_num
            tar_img = load_image_from_path(target_image_path, target_size[0],
                                           target_size[1])
        else:
            img = load_image_from_path(image_path, input_size)
            tar_img = load_image_from_path(target_image_path, input_size)
            bls_h = bls_w = bls
            target_size = [input_size, input_size]
        output_image = np.zeros([target_size[1], target_size[0], 3])
        config = init(epsilon=epsilon, steps=steps, mode=mode, rate=rate)
        config['parameters']["input_size"] = bls

        if mask:
            mask_path = args.mask_path
            processed_mask = load_image_from_path(mask_path, target_size[0], target_size[1])
        else:
            processed_mask = None
        for i in tqdm(range(block_num)):
            for j in tqdm(range(block_num)):
                if processed_mask is not None:
                    input_mask = Image.fromarray(np.array(processed_mask)[bls_w*i: bls_w*i+bls_w, bls_h*j: bls_h*j + bls_h])
                else:
                    input_mask = None
                img_block = Image.fromarray(np.array(img)[bls_w*i: bls_w*i+bls_w, bls_h*j: bls_h*j + bls_h])
                tar_block = Image.fromarray(np.array(tar_img)[bls_w*i: bls_w*i+bls_w, bls_h*j: bls_h*j + bls_h])

                output_image[bls_w*i: bls_w*i+bls_w, bls_h*j: bls_h*j + bls_h] = infer(img_block, config, tar_block, input_mask)

        output = Image.fromarray(output_image.astype(np.uint8))
        output_name = os.path.join('outputs/images', args.output_name)
        save_parameter = '_' + str(epsilon) + '_' + str(steps) + '_' + str(input_size) + '_' + str(block_num) + '_' + str(mode) + '_' + str(args.rate) + '_' + str(int(mask)) + '_' + str(int(resize))
        output_name += save_parameter + '.png'
        print("Output image saved in path {}".format(output_name))
        output.save(output_name)