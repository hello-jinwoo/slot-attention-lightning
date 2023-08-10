import math
import random
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

def get_transforms(
    transform_content: str = 'translate',
    img_size: int = 128, 
    crop_size: int = 192,
    template_size: int = 240,
    max_num_masks: int = 7, 
):
    # structure of insts (K, 8)
    #   - each instruction is composed of 5 elements.. (as concatenated list for preparing batch)
    #       1. [0:2] position of target object : 2D (x, y)
    #       2. [2:4] translation value : 2D (t_x, t_y), default 0
    #       3. [4:5] scaling value : 1D (s), default 1
    #       4. [5:8] color tinting value : 3D (h, s, l), default 0, 1, 1

    insts = torch.zeros(max_num_masks, 8)
    # TODO: patch-level instruction -> max_num_objs

    # set default values
    insts[:, 0:2] = 0.5 # obj_pos
    insts[:, 2:4] = 0.0 # translate
    insts[:, 4:5] = 1.0 # scale
    insts[:, 5:6] = 0.0 # hue
    insts[:, 6:8] = 1.0 # saturation and lightness

    orig_transform = False # the below code assumes orig_transform == False

    transform_ori = None 
    transform_aug = None 

    if crop_size < 0: # random crop mode
        crop_size = random.randint(160, 224)

    d_ori, d_aug = torch.zeros(2,), torch.zeros(2,)
    s1, s2 = 1., 1.
    theta = 0.

    # translate
    if 'translate' in transform_content:
        d_max = (template_size - crop_size) // 2
        if orig_transform:
            d_ori = torch.randint(-d_max, d_max, (2,)) 
        else: 
            d_ori = torch.zeros(2,)
        d_aug = torch.randint(-d_max, d_max, (2,))
    else:
        d_ori = torch.zeros(2,)
        d_aug = torch.zeros(2,)

    # scale
    if 'scale' in transform_content:
        scale_min = crop_size / template_size
        scale_max = template_size / crop_size
        s1, s2 = torch.FloatTensor(2,).uniform_(scale_min, scale_max)
        s1, s2 = float(s1), float(s2)
        if not orig_transform:
            s1 = 1
    else:
        s1, s2 = 1, 1

    insts[:, 2:4] = (d_aug - d_ori) / crop_size # translate (x, y) # TODO: -1 needed? 
    insts[:, 4:5] = s2 / s1 # scale 

    # apply transforms
    transform_ori = [
        transforms.Lambda(lambda img: transforms.functional.center_crop(img=img, output_size=template_size//s1)), 
        transforms.Resize(template_size),
        transforms.Lambda(lambda img: transforms.functional.affine(img=img, angle=0, translate=(d_ori[0], d_ori[1]), scale=1, shear=0)),
        # transforms.Lambda(lambda img: transforms.functional.crop(img=img, top=ny_ori, left=nx_ori, height=img_size, width=img_size)),
        transforms.Lambda(lambda img: transforms.functional.center_crop(img=img, output_size=crop_size)),
        transforms.Resize(img_size),
    ]
    # TODO: note that this process assumes there are only one-content changes (no composite contents changes)
    transform_aug = [
        transforms.Lambda(lambda img: transforms.functional.center_crop(img=img, output_size=template_size//s2)),
        transforms.Resize(template_size),
        transforms.Lambda(lambda img: transforms.functional.affine(img=img, angle=0, translate=(d_aug[0], d_aug[1]), scale=1, shear=0)),
        # transforms.Lambda(lambda img: transforms.functional.crop(img=img, top=ny_aug, left=nx_aug, height=img_size, width=img_size)),
        transforms.Lambda(lambda img: transforms.functional.center_crop(img=img, output_size=crop_size)),
        transforms.Resize(img_size),
    ]

    transform_ori = transforms.Compose(transform_ori)
    transform_aug = transforms.Compose(transform_aug)

    return insts, transform_ori, transform_aug

def get_inv_insts(insts):
    inv_insts = insts.clone()
    inv_insts[:, 2:4] = -insts[:, 2:4]  # translate
    inv_insts[:, 4:5] = 1/insts[:, 4:5] # scale
    inv_insts[:, 5:6] = -insts[:, 5:6]  # hue
    inv_insts[:, 6:8] = 1/insts[:, 6:8]   # saturation and lightness
    return inv_insts

def color_transform(img, insts):
    insts = insts.clone()
    img = img[None, ...] # (1, 3, H, W)
    # rgb2hsl
    img_hsl = rgb2hsl_torch(img) # (1, 3, H, W)

    h = random.uniform(-1, 1)
    s = math.exp(random.uniform(-1, 1)) # TODO: make it configurable
    # l = random.uniform(0.5, 1.5) # TODO: make it configurable
    
    insts[:, 5] = h
    insts[:, 6] = s
    # insts[:, 8] = l

    # TODO: When we change s and l, there will be some saturated pixels
    #       for that reason, now only the h value is controlled.
    #       To handle all the rgb, manipulating s and l is needed
    
    # modify h
    img_hsl[:, 0, ...] = img_hsl[:, 0, ...] + h # manipulate hue (h)
    # if the value is bigger than 1, subtract 1 so that the h values range [0, 1]
    img_hsl[:, 0, ...] = img_hsl[:, 0, ...] - (img_hsl[:, 0, ...] > 1).float() 
    # if the value is smaller than 0, add 1 so that the h values range [0, 1]
    img_hsl[:, 0, ...] = img_hsl[:, 0, ...] + (img_hsl[:, 0, ...] < 0).float() 
    # modify s and l
    img_hsl[:, 1, ...] = torch.clip(img_hsl[:, 1, ...] * s, 0., 1.) # 0~1
    # img_hsl[:, 2, ...] = img_hsl[:, 2, ...] * l

    # clip
    img_hsl = torch.clamp(img_hsl, 0, 1)

    # hsl2rgb
    img_rgb = hsl2rgb_torch(img_hsl) # (1, 3, H, W)

    return img_rgb[0], insts

def rgb2hsl_torch(rgb: torch.Tensor) -> torch.Tensor:
    ''' src: https://github.com/limacv/RGB_HSV_HSL '''
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsl_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsl_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsl_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsl_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsl_h[cmax_idx == 3] = 0.
    hsl_h /= 6.

    hsl_l = (cmax + cmin) / 2.
    hsl_s = torch.empty_like(hsl_h)
    hsl_s[hsl_l == 0] = 0
    hsl_s[hsl_l == 1] = 0
    hsl_l_ma = torch.bitwise_and(hsl_l > 0, hsl_l < 1)
    hsl_l_s0_5 = torch.bitwise_and(hsl_l_ma, hsl_l <= 0.5)
    hsl_l_l0_5 = torch.bitwise_and(hsl_l_ma, hsl_l > 0.5)
    hsl_s[hsl_l_s0_5] = ((cmax - cmin) / (hsl_l * 2.))[hsl_l_s0_5]
    hsl_s[hsl_l_l0_5] = ((cmax - cmin) / (- hsl_l * 2. + 2.))[hsl_l_l0_5]
    return torch.cat([hsl_h, hsl_s, hsl_l], dim=1)


def rgb2hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
    ''' src: https://github.com/limacv/RGB_HSV_HSL '''
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


def hsv2rgb_torch(hsv: torch.Tensor) -> torch.Tensor:
    ''' src: https://github.com/limacv/RGB_HSV_HSL '''
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb


def hsl2rgb_torch(hsl: torch.Tensor) -> torch.Tensor:
    ''' src: https://github.com/limacv/RGB_HSV_HSL '''
    hsl_h, hsl_s, hsl_l = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]
    _c = (-torch.abs(hsl_l * 2. - 1.) + 1) * hsl_s
    _x = _c * (-torch.abs(hsl_h * 6. % 2. - 1) + 1.)
    _m = hsl_l - _c / 2.
    idx = (hsl_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsl)
    _o = torch.zeros_like(_c)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb