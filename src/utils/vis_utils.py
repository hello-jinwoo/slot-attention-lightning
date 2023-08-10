import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import matplotlib.pyplot as plt

def show(imgs):
    plt.figure(figsize=(10, 5))
    plt.rcParams["savefig.bbox"] = 'tight'
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torch.clamp(img, min=0.0, max=1.0)
        img = VF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def visualize(image, recon_combined, recons, pred_masks, gt_masks, attns, colored_box=True):
    """
    `image`: [B, 3, H, W]
    `recon_combined`: [B, 3, H, W]
    `recons`: [B, K, H, W, C]
    `pred_masks`: [B, K, H, W, 1]
    `gt_masks`: [B, K, H, W, 1]
    `attns`: (B, K, N_heads, N_in)
    """

    img = image[:1]
    recon_combined = recon_combined[:1]
    recons = recons[:1]
    pred_masks = pred_masks[:1]
    gt_masks = gt_masks[:1]
    attns = attns[:1]

    _, K, H, W, _ = pred_masks.shape

    # get binarized masks
    pred_mask_max_idxs = torch.argmax(pred_masks.squeeze(-1), dim=1)
    # `mask_max_idxs`: (1, H, W)

    pred_seg_masks = torch.zeros_like(pred_masks.squeeze(-1))
    pred_seg_masks[
        torch.arange(1)[:, None, None],
        pred_mask_max_idxs,
        torch.arange(H)[None, :, None],
        torch.arange(W)[None, None, :],
    ] = 1.0
    pred_seg_masks = pred_seg_masks.unsqueeze(-1)
    # `pred_seg_masks`: (1, K, H, W, 1)

    pad = (0, 0, 2, 2, 2, 2)

    # set colors
    slot_colors = (
        torch.tensor(
            np.array(
                [
                    [255, 0, 0],
                    [255, 127, 0],
                    [255, 255, 0],
                    [0, 255, 0],
                    [0, 0, 255],
                    [75, 0, 130],
                    [148, 0, 211],
                    [0, 255, 255],
                    [153, 255, 153],
                    [255, 153, 204],
                    [102, 0, 51],
                    [128, 128, 128],
                    [255, 255, 255],
                ]
            ),
            dtype=torch.float32,
        )
        / 255.0
    )

    # handle the multi-head attention
    attns = torch.mean(attns, dim=2).view(1, K, H, W).unsqueeze(-1)
    # `attns`: (1, K, H, W, 1)

    attns = torch.cat([attns, pred_masks, pred_seg_masks], dim=0)
    N_row = attns.shape[0]
    # `attns`: (N_row, K, H, W, 1)
    # N_row = T + 2
    # `attns` - attention maps over iterations
    # `pred_masks` - alpha mask generated by decoder
    # `pred_seg_masks` - binary mask from `pred_masks` with argmax

    img = torch.einsum("nchw->nhwc", img)
    gt_col = torch.ones((N_row, H, W, 3), dtype=img.dtype, device=img.device)
    gt_col[-2] = 0  # to draw seg mask by adding values
    gt_col[-1] = img
    # draw boundary box for the original image
    gt_col = F.pad(gt_col, pad=pad, mode="constant", value=1.0)
    if colored_box:
        gt_col[1:, :2, :] = gt_col[1:, -2:, :] = gt_col[1:, :, :2] = gt_col[
            1:, :, -2:
        ] = slot_colors[
            -2
        ]  # gray
    gt_col = F.pad(gt_col, pad=pad, mode="constant", value=1.0)
    # `gt_col`: [T+2, H, W, C]

    recon_combined = torch.einsum("nchw->nhwc", recon_combined)
    pred_col = torch.ones(
        (N_row, H, W, 3), dtype=recon_combined.dtype, device=recon_combined.device
    )
    pred_col[-2] = 0  # to draw seg mask by adding values
    pred_col[-1] = recon_combined

    # draw boundary box for the reconstructed image
    pred_col = F.pad(pred_col, pad=pad, mode="constant", value=1.0)
    if colored_box:
        pred_col[1:, :2, :] = pred_col[1:, -2:, :] = pred_col[1:, :, :2] = pred_col[
            1:, :, -2:
        ] = slot_colors[
            -2
        ]  # gray
    pred_col = F.pad(pred_col, pad=pad, mode="constant", value=1.0)
    # `pred_col`: [T+2, H, W, C]

    for k in range(K):

        # # get vis. of attention maps based on the original image
        picture = torch.ones_like(img) * attns[:, k, :, :, :]

        # overwrite vis. of alpha mask with the recon. image
        picture[-2] = pred_seg_masks[:, k, :, :, :]
        picture[-1] = recons[:, k, :, :, :]
        # picture[-1] = recons[:, k, :, :, :] * pred_seg_masks[:, k, :, :, :] + (
        #     1 - pred_seg_masks[:, k, :, :, :]
        # )

        try:
            gt_col[-2, 4:-4, 4:-4, :] += gt_masks[0, k, :, :, :] * slot_colors[k - 1].to(
                pred_seg_masks.device
            )  # `k-1` -> to give white color to background
        except:
            # when #slots > # objects. it is not the big deal
            pass 

        pred_col[-2, 4:-4, 4:-4, :] += pred_seg_masks[0, k, :, :, :] * slot_colors[k - 1].to(
            pred_seg_masks.device
        )  # `k-1` -> to give white color to background

        # draw boundary box for slots
        picture = F.pad(picture, pad=pad, mode="constant", value=1.0)
        if colored_box:
            picture[:, :2, :] = picture[:, -2:, :] = picture[:, :, :2] = picture[
                :, :, -2:
            ] = slot_colors[k]
        picture = F.pad(picture, pad=pad, mode="constant", value=1.0)

        if k == 0:
            log_img = torch.cat([picture], dim=2)
        else:
            log_img = torch.cat([log_img, picture], dim=2)

    bg_mask = torch.where(
        torch.sum(gt_col[-2, 4:-4, 4:-4, :], dim=-1, keepdim=True) == 0,
        torch.ones_like(gt_col[-2, 4:-4, 4:-4, :]),
        torch.zeros_like(gt_col[-2, 4:-4, 4:-4, :]),
    )
    # gt_col[-2, 4:-4, 4:-4, :] += bg_mask * 0.5
    log_img = torch.cat([gt_col, pred_col, log_img], dim=2)

    log_img = log_img.permute(0, 3, 1, 2)
    return log_img