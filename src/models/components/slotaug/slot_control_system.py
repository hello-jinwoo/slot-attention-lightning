import torch 
import torch.nn as nn 
from torchvision.utils import make_grid
from src.utils.vis_utils import show, visualize
from src.models.components.slotaug.slotaug_utils import * 
from src.models.components.slotaug.slotaug_ae import SlotAugAutoEncoder
from src.models.components.slota.slota_ae import SlotAttentionAutoEncoder

class SlotControlSystem():

    def __init__(
        self, 
        model_name=None, 
        model_config=None, 
        ckpt_path=None, 
        device=torch.device('cpu')
    ): 
        """ ATTRIBUTES """
        self.model_name = None  # ["slotaug", "slota"]
        self.model = None  
        self.ckpt = None 

        self.img_size = 128
        self.num_slots = 7 # fix it to 7? otherwise, we should modify the model (self.scale)
        self.num_valid_slots = 0 # 
        self.pure_slots = torch.tensor([])      # (1, N_slots, D_slot), before apply_insts()
        self.slots_ori = torch.tensor([])           # (1, N_slots, D_slot), after apply_insts()
        self.slots = torch.tensor([])           # (1, N_slots, D_slot), after apply_insts()
        self.centered_slots_ori = torch.tensor([])  # (1, N_slots, D_slot)
        self.centered_slots = torch.tensor([])  # (1, N_slots, D_slot)
        self.attns_ori = torch.tensor([])           # (1, N_slots, N_heads, N_in)
        self.normed_attns_ori = torch.tensor([])     # (1, N_slots, N_heads, N_in)
        self.attns = torch.tensor([])           # (1, N_slots, N_heads, N_in)
        self.normed_attns = torch.tensor([])     # (1, N_slots, N_heads, N_in)
        
        self.obj_pos_ori = torch.tensor([])             # TODO:
        self.obj_pos = torch.tensor([])                 # TODO: (1, N_slots, 2), obj pos corresponding to self.slots? pure_slots?
        self.obj_pos_filtered_ori = torch.tensor([])    # TODO: after no obj filtering, (1, N_slots, 2),
        self.obj_pos_filtered = torch.tensor([])        # TODO: after no obj filtering, (1, N_slots, 2),
        # self.valid_slot_idxs = torch.tensor([]) # TODO: (1, N_valid_slots,)
        self.valid_slots_mask_ori = torch.tensor([]) # (1, N_slots,)
        self.valid_slots_mask = torch.tensor([]) # (1, N_slots,)
        self.background_mask_ori = torch.tensor([]) # (1, N_slots,)
        self.background_mask = torch.tensor([]) # (1, N_slots,)
        self.obj_exist_mask_ori = torch.tensor([])  # (1, N_slots,) except background from valid_slots_mask
        self.obj_exist_mask = torch.tensor([])  # (1, N_slots,) except background from valid_slots_mask

        self.img_input = torch.tensor([])       # (1, 3, H, W)
        
        self.recons_ori = torch.tensor([])  # (1, K, H, W, 3)
        self.recons = torch.tensor([])      # (1, K, H, W, 3)
        self.masks_ori = torch.tensor([])   # (1, K, H, W, 1)
        self.masks = torch.tensor([])       # (1, K, H, W, 1)
        self.recon_combined_ori = torch.tensor([])     # (1, 3, H, W)
        self.recon_combined = torch.tensor([])         # (1, 3, H, W)

        self.target_slot = torch.tensor([])     # (1, D_slot) # TODO
        self.dist_metric = "mse" # ["mse", "rmse", "cosine"]

        self.dists_ori = torch.zeros(1, self.num_slots)
        self.dists_prev = torch.zeros(1, self.num_slots)
        self.dists_centered_ori = torch.zeros(1, self.num_slots)
        self.dists_centered_prev = torch.zeros(1, self.num_slots)
        self.norms_ori = torch.zeros(1, self.num_slots)
        self.norms = torch.zeros(1, self.num_slots)
        self.norms_centered_ori = torch.zeros(1, self.num_slots)
        self.norms_centered = torch.zeros(1, self.num_slots)

        self.insts_dim = 8
        self.aux_identity_insts = torch.zeros(1, 1, self.insts_dim, dtype=torch.float32)
        self.aux_identity_insts[:, :, 4:5] = 1 # scale
        self.aux_identity_insts[:, :, 6:8] = 1 # saturation and lightness
        # [:, :, 6] is for h (hue) so the default value == 0
        self.device = device

        self.model_name= model_name 
        self.model_config = model_config
        self.ckpt_path = ckpt_path 
        if model_config: 
            self.set_model(model_config, ckpt_path)

        """ DURA """
        self.dura_rmse = list()
        self.dura_mse = list()
        self.dura_cosine = list()
        self.dura_pos = list()


    """ ATTRIBUTE HANDLERS """
    def set_model(
        self, 
        model_config: dict = None,
        ckpt_path: str = None,
    ):
        if self.model_name == "slotaug":
            self.model = SlotAugAutoEncoder(**model_config).to(self.device)
        elif self.model_name == "slota":
            self.model = SlotAttentionAutoEncoder(**model_config).to(self.device)
        self.model.eval()
        if ckpt_path: 
            self.set_ckpt(ckpt_path)

    def set_ckpt(
        self, 
        ckpt_path: str = None,
    ):
        self.ckpt_path = ckpt_path 
        self.ckpt = dict() 
        for k, v in torch.load(self.ckpt_path)['state_dict'].items():
            k = k.replace('net.', '', 1)
            if hasattr(self.model, "decoder") and "decoder_cnn." in k:
                k = k.replace("decoder_cnn", "decoder")
            elif hasattr(self.model, "decoder_cnn") and "decoder." in k:
                k = k.replace("decoder", "decoder_cnn")

            if hasattr(self.model, "encoder") and "encoder_cnn." in k:
                k = k.replace("encoder_cnn", "encoder")
            elif hasattr(self.model, "encoder_cnn") and "encoder." in k:
                k = k.replace("encoder", "encoder_cnn")

            self.ckpt[k] = v
        self.model.load_state_dict(self.ckpt)
        self.model.eval()

    # TODO: as a model param or on-the-fly param?
    def set_num_slots(
        self,
        num_slots
    ):
        self.num_slots = num_slots
        self.model.num_slots = num_slots 
        if self.model_name == "slotaug":
            self.model.slotaug.num_slots = num_slots 
        elif self.model_name == "slota": 
            self.model.slot_attention.num_slots = num_slots

    # TODO: necessary?
    def set_img_size( 
        self, 
        img_size: int = 128, 
    ): 
        self.img_size = img_size 

    def reset(
        self,
    ):
        self.slots = self.slots_ori
        self.centered_slots = self.centered_slots_ori
        self.attns = self.attns_ori
        self.normed_attns = self.normed_attns_ori
        
        self.obj_pos = self.obj_pos_ori
        self.valid_slots_mask = self.valid_slots_mask_ori
        self.background_mask = self.background_mask_ori
        self.obj_exist_mask = self.obj_exist_mask_ori
        self.obj_pos_filtered = self.obj_pos_filtered_ori
        # self.valid_slot_idxs = self.valid_slot_idxs_ori
        self.num_valid_slots = self.num_valid_slots_ori
        
        self.recons = self.recons_ori
        self.masks = self.masks_ori
        self.recon_combined = self.recon_combined_ori

        self.dists_ori = torch.zeros(1, self.num_slots)
        self.dists_prev = torch.zeros(1, self.num_slots)
        self.dists_centered_ori = torch.zeros(1, self.num_slots)
        self.dists_centered_prev = torch.zeros(1, self.num_slots)
        self.norms = self.norms_ori
        self.norms_centered = self.norms_centered_ori

        self.set_dist_metric() # rmse, mse, cosine

    """ SLOT MANIPULATION """
    def get_insts(
        self,
        num_insts: int = 0,
        target_pos: torch.tensor = torch.tensor([]), 
        translate: torch.tensor = torch.tensor([]), 
        scale: torch.tensor = torch.tensor([]), 
        color: torch.tensor = torch.tensor([]), 
        rotate: torch.tensor = torch.tensor([]), 
    ):
        # initialize insts
        insts = torch.zeros(1, self.num_slots, 8, dtype=torch.float32) # TODO: num_slots -> num_insts
        insts[0, :, 2:4] = 0    # translation value : 2D (t_x, t_y), default 0
        insts[0, :, 4:5] = 1.0  # scaling value : 1D (s), default 1
        insts[0, :, 5:6] = 0.0  # h
        insts[0, :, 6:8] = 1.0  # l, s
        if rotate.nelement() > 0:
            insts = torch.cat([insts, torch.zeros(1, self.num_slots, 1)], dim=-1)
        
        # update values
        if num_insts == -1:
            insts[0, :, 0:2] = target_pos[0] if target_pos.nelement() > 0 else 0.5
            insts[0, :, 2:4] = translate[0] if translate.nelement() > 0 else 0.
            insts[0, :, 4:5] = scale[0] if scale.nelement() > 0 else 1.
            insts[0, :, 5:8] = color[0] if color.nelement() > 0 else torch.tensor([0., 1., 1.])
        else:
            for inst_idx in range(num_insts):
                insts[0, inst_idx, 0:2] = target_pos[inst_idx] if target_pos.nelement() > 0 else 0.5
                insts[0, inst_idx, 2:4] = translate[inst_idx] if translate.nelement() > 0 else 0.
                insts[0, inst_idx, 4:5] = scale[inst_idx] if scale.nelement() > 0 else 1.
                insts[0, inst_idx, 5:8] = color[inst_idx] if color.nelement() > 0 else torch.tensor([0., 1., 1.])

        return insts
    
    def get_inv_insts(
        self, 
        insts: torch.tensor = torch.tensor([]), 
    ):
        # data_utils.get_inv_insts()
        return get_inv_insts(insts)

    def apply_insts_batch(
        self, 
        insts: torch.tensor = torch.tensor([]), # (1, K, 11)
        slots: torch.tensor = torch.tensor([]), # (1, K, D_slots)
        centered_slots: torch.tensor = torch.tensor([]), # (1, K, D_slots)
        obj_pos: torch.tensor = torch.tensor([]), # (1, K, 2)
        inplace: bool = False, 
        visualize: str = "recons", # ["", "recon_combined", "recons"]
        verbose: bool = False, 
    ):
        if slots.nelement() == 0:
            slots = self.slots 
            centered_slots = self.centered_slots
            obj_pos = self.obj_pos_filtered
        elif centered_slots.nelement() == 0:
            centered_slots = torch.zeros_like(self.slots)

        slots_prev = slots
        centered_slots_prev = centered_slots
        obj_pos_prev = obj_pos
        insts = self.rearrange_insts(insts, obj_pos) # change the order of insts to be matched with the closest slot
        slots = self.model.slotaug.apply_insts(insts, slots)

        # get obj pos from alphamask
        recons, masks, recon_combined = self.scene_generation(slots, visualize=visualize) # `masks`: (1, N_slots, H, W, 1)
        attns, normed_attns = masks_to_attns(masks)
        obj_pos = self.get_obj_pos(normed_attns)
        obj_pos_filtered, valid_slots_mask  = self.filter_no_obj(obj_pos, attns, replace_value=-2.0) # -2 for no_obj
        obj_pos_filtered, background_mask  = self.filter_background(obj_pos_filtered, attns, replace_value=2.0) # 2 for background
        obj_exist_mask = valid_slots_mask - background_mask # (1, N_slots) # TODO: sanity check whether there is no case of valid_slots_mask not including background_mask
        num_valid_slots = torch.sum(valid_slots_mask).item()

        centered_slots = self.get_centered_slots(slots=slots, obj_pos=obj_pos_filtered, obj_mask=obj_exist_mask)
        
        dists_ori = self.get_slot_dist(slots_src=self.slots_ori, obj_pos_src=self.obj_pos_filtered_ori, slots_tgt=slots, obj_pos_tgt=obj_pos_filtered)
        dists_prev = self.get_slot_dist(slots_src=slots_prev, obj_pos_src=obj_pos_prev, slots_tgt=slots, obj_pos_tgt=obj_pos_filtered)
        dists_centered_ori = self.get_slot_dist(
            slots_src=self.centered_slots_ori, obj_pos_src=self.obj_pos_filtered_ori,
            slots_tgt=centered_slots, obj_pos_tgt=obj_pos_filtered
        )
        dists_centered_prev = self.get_slot_dist(
            slots_src=centered_slots_prev, obj_pos_src=obj_pos_prev,
            slots_tgt=centered_slots, obj_pos_tgt=obj_pos_filtered
        )

        norms_prev = self.get_slot_norm(slots=slots_prev)
        norms_centered_prev = self.get_slot_norm(slots=centered_slots_prev)
        norms = self.get_slot_norm(slots=slots)
        norms_centered = self.get_slot_norm(slots=centered_slots)

        if verbose:
            norm = torch.mean(norms).item()
            print("norms: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*norms[0].tolist()))
            print("norm (mean): {:.4f}".format(norm))

            # norm_ori_dists = torch.sqrt((norms - self.norms_ori) ** 2)
            # norm_ori_dist = torch.mean(norm_ori_dists).item()
            # print("norm_ori_dists: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*norm_ori_dists[0].tolist()))
            # print("norm_ori_dist (mean): {:.4f}".format(norm_ori_dist))

            # norm_prev_dists = torch.sqrt((norms - norms_prev) ** 2)
            # norm_prev_dist = torch.mean(norm_prev_dists).item()
            # print("norm_prev_dists: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*norm_prev_dists[0].tolist()))
            # print("norm_prev_dist (mean): {:.4f}".format(norm_prev_dist))

            # norm_centered_dists_ori = torch.sqrt((norms - self.norms_centered_ori) ** 2)
            # norm_centered_dist_ori = torch.mean(norm_centered_dists_ori).item()
            # print("norm_centered_dists_ori: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*norm_centered_dists_ori[0].tolist()))
            # print("norm_centered_dist_ori (mean): {:.4f}".format(norm_centered_dist_ori))
            
            # norm_centered_dists_prev = torch.sqrt((norms_centered - norms_centered_prev) ** 2)
            # norm_centered_dist_prev = torch.mean(norm_centered_dists_prev).item()
            # print("norm_centered_dists_prev: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*norm_centered_dists_prev[0].tolist()))
            # print("norm_centered_dist_prev (mean): {:.4f}".format(norm_centered_dist_prev))

            dist_ori = torch.mean(dists_ori).item()
            print("dists_ori: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*dists_ori[0].tolist()))
            print("dist_ori (mean): {:.4f}".format(dist_ori))

            # dist_prev = torch.mean(dists_prev).item()
            # print("dists_prev: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*dists_prev[0].tolist()))
            # print("dist_prev (mean): {:.4f}".format(dist_prev))

            dist_centered_ori = torch.mean(dists_centered_ori).item()
            print("dists_centered_ori: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*dists_centered_ori[0].tolist()))
            print("dist_centered_ori (mean): {:.4f}".format(dist_centered_ori))

            # dist_centered_prev = torch.mean(dists_centered_prev).item()
            # print("dists_centered_prev: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*dists_centered_prev[0].tolist()))
            # print("dist_centered_prev (mean): {:.4f}".format(dist_centered_prev))

        if inplace: 
            self.slots = slots
            self.centered_slots = centered_slots
            self.attns = attns
            self.normed_attns = normed_attns
            self.obj_pos = obj_pos 
            self.obj_pos_filtered = obj_pos_filtered 
            self.valid_slots_mask = valid_slots_mask
            self.background_mask = background_mask
            self.obj_exist_mask = obj_exist_mask
            # self.valid_slot_idxs = None # TODO
            self.num_valid_slots = num_valid_slots
            self.recons = recons
            self.masks = masks 
            self.recon_combined = recon_combined

            self.dists_ori = dists_ori
            self.dists_prev = dists_prev
            self.dists_centered_ori = dists_centered_ori
            self.dists_centered_prev = dists_centered_prev

            self.norms = norms
            self.norms_centred = norms_centered

        return  {
            "slots": slots,
            "centered_slots": centered_slots, 
            "attns": attns, 
            "normed_attns": normed_attns, 
            "obj_pos": obj_pos, 
            "obj_pos_filtered": obj_pos_filtered,
            "obj_exist_mask": obj_exist_mask,
            # "valid_slot_idxs": None, # TODO
            "num_valid_slots": num_valid_slots,
            "recons": recons,
            "masks": masks,  
            "recon_combined": recon_combined, 

            "dists_ori": dists_ori,
            "dists_prev": dists_prev,
            "dists_centered_ori": dists_centered_ori,
            "dists_centered_prev": dists_centered_prev,
            
            "norms_prev": norms_prev,
            "norms_centered_prev": norms_centered_prev,
            "norms": norms,
            "norms_centered": norms_centered,
        }

    def apply_inst_individual(
        self, 
        slot_idx=None, # integer if input exists
        target_obj_pos: torch.tensor = torch.tensor([]), # (1, 1, 2) 
        inst: torch.tensor = torch.tensor([]), # (1, 1, 11)
        slots: torch.tensor = torch.tensor([]), # (1, K, D_slot)
        centered_slots: torch.tensor = torch.tensor([]), # (1, K, D_slot)
        obj_pos: torch.tensor = torch.tensor([]), # (1, K, 2) pos of objs in a image
        inplace: bool = False, 
        visualize: str = "recons", # ["", "recon_combined", "recons"]
        verbose: bool = False, 
    ):
        if slots.nelement() == 0:
            slots = self.slots 
            centered_slots = self.centered_slots
            obj_pos = self.obj_pos_filtered
        elif centered_slots.nelement() == 0:
            centered_slots = torch.zeros_like(self.slots)
    
        if slot_idx == None:
            pos_dist = torch.sum((obj_pos - target_obj_pos) ** 2, dim=-1) # (1, K)
            slot_idx = torch.argmax(pos_dist) # single value

        slots_prev = slots
        centered_slots_prev = centered_slots
        obj_pos_prev = obj_pos

        if type(slot_idx) == int:
            slot_idx = torch.tensor(slot_idx)
        slots = slots.clone()
        slots[:, slot_idx:slot_idx+1, :] = self.model.slotaug.apply_insts(inst, slots[:, slot_idx:slot_idx+1, :])

        # get obj pos from alphamask
        recons, masks, recon_combined = self.scene_generation(slots, visualize=visualize) # `masks`: (1, N_slots, H, W, 1)
        masks_vis = masks.permute(0, 4, 2, 1, 3).reshape(1, 1, 128, -1)
        masks_vis = torch.cat([torch.zeros(1, 1, 128, 128).to(self.device), masks_vis], dim=-1).repeat(1, 3, 1, 1)
        self.visualize(masks_vis)
        attns, normed_attns = masks_to_attns(masks)
        obj_pos = self.get_obj_pos(normed_attns)
        obj_pos_filtered, valid_slots_mask  = self.filter_no_obj(obj_pos, attns, replace_value=-2.0) # -2 for no_obj
        obj_pos_filtered, background_mask  = self.filter_background(obj_pos_filtered, attns, replace_value=2.0) # 2 for background
        obj_exist_mask = valid_slots_mask - background_mask # (1, N_slots) # TODO: sanity check whether there is no case of valid_slots_mask not including background_mask
        num_valid_slots = torch.sum(valid_slots_mask).item()

        centered_slots = self.get_centered_slots(slots=slots, obj_pos=obj_pos_filtered, obj_mask=obj_exist_mask)
        
        dists_ori = self.get_slot_dist(slots_src=self.slots_ori, obj_pos_src=self.obj_pos_filtered_ori, slots_tgt=slots, obj_pos_tgt=obj_pos_filtered, metric="mse")
        dists_ori_rmse = self.get_slot_dist(slots_src=self.slots_ori, obj_pos_src=self.obj_pos_filtered_ori, slots_tgt=slots, obj_pos_tgt=obj_pos_filtered, metric="rmse")
        dists_ori_cosine = self.get_slot_dist(slots_src=self.slots_ori, obj_pos_src=self.obj_pos_filtered_ori, slots_tgt=slots, obj_pos_tgt=obj_pos_filtered, metric="cosine")
        dists_prev = self.get_slot_dist(slots_src=slots_prev, obj_pos_src=obj_pos_prev, slots_tgt=slots, obj_pos_tgt=obj_pos_filtered)
        dists_centered_ori = self.get_slot_dist(
            slots_src=self.centered_slots_ori, obj_pos_src=self.obj_pos_filtered_ori,
            slots_tgt=centered_slots, obj_pos_tgt=obj_pos_filtered
        )
        dists_centered_prev = self.get_slot_dist(
            slots_src=centered_slots_prev, obj_pos_src=obj_pos_prev,
            slots_tgt=centered_slots, obj_pos_tgt=obj_pos_filtered
        )

        norms_prev = self.get_slot_norm(slots=slots_prev)
        norms_centered_prev = self.get_slot_norm(slots=centered_slots_prev)
        norms = self.get_slot_norm(slots=slots)
        norms_centered = self.get_slot_norm(slots=centered_slots)

        if verbose:
            norm = torch.mean(norms).item()
            print("norms: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*norms[0].tolist()))
            print("norm (mean): {:.4f}".format(norm))

            # norm_ori_dists = torch.sqrt((norms - self.norms_ori) ** 2)
            # norm_ori_dist = torch.mean(norm_ori_dists).item()
            # print("norm_ori_dists: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*norm_ori_dists[0].tolist()))
            # print("norm_ori_dist (mean): {:.4f}".format(norm_ori_dist))

            # norm_prev_dists = torch.sqrt((norms - norms_prev) ** 2)
            # norm_prev_dist = torch.mean(norm_prev_dists).item()
            # print("norm_prev_dists: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*norm_prev_dists[0].tolist()))
            # print("norm_prev_dist (mean): {:.4f}".format(norm_prev_dist))

            # norm_centered_dists_ori = torch.sqrt((norms_centered - self.norms_centred_ori) ** 2)
            # norm_centered_dist_ori = torch.mean(norm_centered_dists_ori).item()
            # print("norm_centered_dists_ori: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*norm_centered_dists_ori[0].tolist()))
            # print("norm_centered_dist_ori (mean): {:.4f}".format(norm_centered_dist_ori))
            
            # norm_centered_dists_prev = torch.sqrt((norms_centered - norms_centered_prev) ** 2)
            # norm_centered_dist_prev = torch.mean(norm_centered_dists_prev).item()
            # print("norm_centered_dists_prev: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*norm_centered_dists_prev[0].tolist()))
            # print("norm_centered_dist_prev (mean): {:.4f}".format(norm_centered_dist_prev))

            dist_ori = torch.mean(dists_ori).item()
            print("dists_ori: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*dists_ori[0].tolist()))
            print("dist_ori (mean): {:.4f}".format(dist_ori))

            dist_ori_rmse = torch.mean(dists_ori_rmse).item()
            print("dists_ori_rmse: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*dists_ori[0].tolist()))
            print("dist_ori_rmse (mean): {:.4f}".format(dist_ori_rmse))

            dist_ori_cosine = torch.mean(dists_ori_cosine).item()
            print("dists_ori_cosine: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*dists_ori[0].tolist()))
            print("dist_ori_cosine (mean): {:.4f}".format(dist_ori_cosine))

            # dist_prev = torch.mean(dists_prev).item()
            # print("dists_prev: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*dists_prev[0].tolist()))
            # print("dist_prev (mean): {:.4f}".format(dist_prev))

            dist_centered_ori = torch.mean(dists_centered_ori).item()
            print("dists_centered_ori: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*dists_centered_ori[0].tolist()))
            print("dist_centered_ori (mean): {:.4f}".format(dist_centered_ori))

            # dist_centered_prev = torch.mean(dists_centered_prev).item()
            # print("dists_centered_prev: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*dists_centered_prev[0].tolist()))
            # print("dist_centered_prev (mean): {:.4f}".format(dist_centered_prev))

        if inplace: 
            self.slots = slots
            self.centered_slots = centered_slots 
            self.attns = attns
            self.normed_attns = normed_attns
            self.obj_pos = obj_pos 
            self.obj_pos_filtered = obj_pos_filtered 
            self.valid_slots_mask = valid_slots_mask
            self.background_mask = background_mask
            self.obj_exist_mask = obj_exist_mask
            # self.valid_slot_idxs = None # TODO
            self.num_valid_slots = num_valid_slots
            self.recons = recons
            self.masks = masks 
            self.recon_combined = recon_combined

            self.dists_ori = dists_ori
            self.dists_ori_rmse = dists_ori_rmse
            self.dists_ori_cosine = dists_ori_cosine
            self.dists_prev = dists_prev
            self.dists_centered_ori = dists_centered_ori
            self.dists_centered_prev = dists_centered_prev

            self.norms = norms
            self.norms_centred = norms_centered

        return  {
            "slots": slots,
            "centered_slots": centered_slots, 
            "attns": attns, 
            "normed_attns": normed_attns, 
            "obj_pos": obj_pos, 
            "obj_pos_filtered": obj_pos_filtered,
            "obj_exist_mask": obj_exist_mask,
            # "valid_slot_idxs": None, # TODO
            "num_valid_slots": num_valid_slots,
            "recons": recons,
            "masks": masks,  
            "recon_combined": recon_combined, 

            "dists_ori": dists_ori,
            "dists_ori_rmse": dists_ori_rmse,
            "dists_ori_cosine": dists_ori_cosine,
            "dists_prev": dists_prev,
            "dists_centered_ori": dists_centered_ori,
            "dists_centered_prev": dists_centered_prev,

            "norms_prev": norms_prev,
            "norms_centered_prev": norms_centered_prev,
            "norms": norms,
            "norms_centered": norms_centered,
        }

    def rearrange_slots(
        self, 
        slots_src: torch.tensor = torch.tensor([]), 
        obj_pos_src: torch.tensor = torch.tensor([]), 
        obj_pos_tgt: torch.tensor = torch.tensor([]), 
        inplace: bool = False,
    ):
        assert obj_pos_tgt.nelement() > 0, "obj_pos_tgt must be given."

        if slots_src.nelement() == 0:
            slots_src = self.slots
            obj_pos_src = self.obj_pos_filtered
        
        matched_idxs = matching_pairs(obj_pos_src, obj_pos_tgt) # (B, K)
        slots_rearranged = rearrange_by_matched_idx(slots_src, matched_idxs)
        obj_pos_rearranged = rearrange_by_matched_idx(obj_pos_src, matched_idxs)

        if inplace: 
            self.pure_slots = rearrange_by_matched_idx(self.pure_slots, matched_idxs)
            self.slots_ori = rearrange_by_matched_idx(self.slots_ori, matched_idxs)
            self.slots = rearrange_by_matched_idx(self.slots, matched_idxs)
            self.centered_slots_ori = rearrange_by_matched_idx(self.centered_slots_ori, matched_idxs)
            self.centered_slots = rearrange_by_matched_idx(self.centered_slots, matched_idxs)
            self.attns_ori = rearrange_by_matched_idx(self.attns_ori, matched_idxs)
            self.normed_attns_ori = rearrange_by_matched_idx(self.normed_attns_ori, matched_idxs)
            self.attns = rearrange_by_matched_idx(self.attns, matched_idxs)
            self.normed_attns = rearrange_by_matched_idx(self.normed_attns, matched_idxs)
            
            self.obj_pos_ori = rearrange_by_matched_idx(self.obj_pos_ori, matched_idxs)
            self.obj_pos = rearrange_by_matched_idx(self.obj_pos, matched_idxs)
            self.valid_slots_mask_ori = rearrange_by_matched_idx(self.valid_slots_mask_ori, matched_idxs)
            self.valid_slots_mask = rearrange_by_matched_idx(self.valid_slots_mask, matched_idxs)
            self.background_mask_ori = rearrange_by_matched_idx(self.background_mask_ori, matched_idxs)
            self.background_mask = rearrange_by_matched_idx(self.background_mask, matched_idxs)
            self.obj_exist_mask_ori = rearrange_by_matched_idx(self.obj_exist_mask_ori, matched_idxs)
            self.obj_exist_mask = rearrange_by_matched_idx(self.obj_exist_mask, matched_idxs)
            self.obj_pos_filtered_ori = rearrange_by_matched_idx(self.obj_pos_filtered_ori, matched_idxs)
            self.obj_pos_filtered = rearrange_by_matched_idx(self.obj_pos_filtered, matched_idxs)
            # self.valid_slot_idxs_ori = torch.tensor([]) # TODO: (N_valid_slots,)
            # self.valid_slot_idxs = torch.tensor([]) # TODO: (N_valid_slots,)

            self.dists_ori = rearrange_by_matched_idx(self.dists_ori, matched_idxs)
            self.dists_prev = rearrange_by_matched_idx(self.dists_prev, matched_idxs)
            self.dists_centered_ori = rearrange_by_matched_idx(self.dists_centered_ori, matched_idxs)
            self.dists_centered_prev = rearrange_by_matched_idx(self.dists_centered_prev, matched_idxs)
            self.norms_ori = rearrange_by_matched_idx(self.norms_ori, matched_idxs)
            self.norms = rearrange_by_matched_idx(self.norms, matched_idxs)
            self.norms_centered_ori = rearrange_by_matched_idx(self.norms_centered_ori, matched_idxs)
            self.norms_centered = rearrange_by_matched_idx(self.norms_centered, matched_idxs)
            
        return matched_idxs, slots_rearranged, obj_pos_rearranged
        
    def rearrange_insts(
            self, 
            insts: torch.tensor = torch.tensor([]), 
            obj_pos: torch.tensor = torch.tensor([]),
    ): 
        """
        rearrange the order of insts to match with that of the given obj_pos
        """
        return rearrange_insts(insts, obj_pos)
    
    def get_obj_pos(
        self,
        normed_attns: torch.tensor = torch.tensor([]),
    ):
        return get_obj_pos_from_attns(normed_attns) # (B, K, 2)
    
    def filter_background(
        self,
        obj_pos: torch.tensor = torch.tensor([]),
        attns: torch.tensor = torch.tensor([]),
        replace_value: float = 0.5, 
        inplace: bool = False,
    ):
        if obj_pos.nelement() == 0:
            obj_pos = self.obj_pos
            attns = self.attns

        obj_pos_filtered, background_mask = filter_background(obj_pos, attns, replace_value)

        if inplace: 
            self.obj_pos_filtered = obj_pos_filtered 
            self.background_mask = background_mask
            
        return obj_pos_filtered, background_mask

    def filter_no_obj(
        self, 
        obj_pos: torch.tensor = torch.tensor([]),
        attns: torch.tensor = torch.tensor([]),
        no_obj_thrs: float = 0.005, 
        replace_value: float = 0.5, 
    ):
        if obj_pos.nelement() == 0:
            obj_pos = self.obj_pos
            attns = self.attns
        # the below filter_no_obj function from slotaug_utils
        obj_pos_filtered, valid_slots_mask = filter_no_obj(obj_pos, attns, no_obj_thrs, replace_value)
        return obj_pos_filtered, valid_slots_mask

    """ CENTERING """
    def get_centered_slots(
        self, 
        slots: torch.tensor = torch.tensor([]),
        obj_pos: torch.tensor = torch.tensor([]),
        obj_mask: torch.tensor = torch.tensor([]),
        inplace: bool = False,
    ):
        centered_slots = slots.clone()
        indices = torch.nonzero(obj_mask)[:,1]
        
        if indices.nelement() == 0:
            return centered_slots
        
        centers = torch.full((1, self.num_slots, 2), 0.5).to(self.device)
        offsets = centers - obj_pos
        insts = self.aux_identity_insts.repeat(1, self.num_slots, 1).to(self.device)
        insts[:, :, 0:2] = obj_pos
        insts[:, :, 3:5] = offsets
        for slot_idx in indices:
            slot_idx = slot_idx.item()
            centered_slots[:, slot_idx:slot_idx+1, :] = self.model.slotaug.apply_insts(insts[:, slot_idx:slot_idx+1, :], slots[:, slot_idx:slot_idx+1, :])
        if inplace:
            self.centered_slots = centered_slots

        return centered_slots
    
    """ OBJECT DISCOVERY """
    def object_discovery(
        self, 
        img: torch.tensor = torch.zeros((1, 128, 128, 3)),
        inplace: bool = False, 
        visualize: str = "recons", # ["", "recon_combined", "recons"]  
    ): 
        if self.model_name == "slotaug":
            with torch.no_grad():
                img = img.to(self.device)
                if hasattr(self.model, "encoder_cnn"):
                    x = self.model.encoder_cnn(img)
                else: 
                    x = self.model.encoder(img)
                outputs = self.model.slotaug.spatial_binding(x)
                pure_slots = outputs["slots"]
                attns = outputs["attns"]
                normed_attns = outputs["normed_attns"]
                slots = self.model.slotaug.apply_insts(
                    insts=self.aux_identity_insts.repeat(1, self.num_slots, 1).to(img.device), 
                    slots=outputs["slots"],
                )
                recons, masks, recon_combined = self.scene_generation(slots, visualize=visualize)
                attns, normed_attns = masks_to_attns(masks)
                obj_pos = self.get_obj_pos(normed_attns) # get obj pos from alpha mask 
                obj_pos_filtered, valid_slots_mask  = self.filter_no_obj(obj_pos, attns, replace_value=-2.0) # -2 for no_obj
                obj_pos_filtered, background_mask  = self.filter_background(obj_pos_filtered, attns, replace_value=2.0) # 2 for background
                obj_exist_mask = valid_slots_mask - background_mask # (1, N_slots) # TODO: sanity check whether there is no case of valid_slots_mask not including background_mask
                num_valid_slots = torch.sum(valid_slots_mask).item()
                centered_slots = self.get_centered_slots(slots, obj_pos_filtered, obj_exist_mask)

                norms = self.get_slot_norm(slots=slots)   # (1, K)
                norms_centered = self.get_slot_norm(slots=centered_slots)
            
            outputs["pure_slots"] = pure_slots 
            outputs["slots"] = slots
            outputs["centered_slots"] = centered_slots
            outputs["attns"] = attns 
            outputs["normed_attns"] = normed_attns
            outputs["obj_pos"] = obj_pos 
            outputs["obj_pos_filtered"] = obj_pos_filtered 
            outputs["valid_slots_mask"] = valid_slots_mask 
            outputs["background_mask"] = background_mask
            outputs["obj_exist_mask"] = obj_exist_mask
            # outputs["valid_slot_idxs"] = None # TODO
            outputs["num_valid_slots"] = num_valid_slots
            outputs["recons"] = recons
            outputs["masks"] = masks
            outputs["recon_combined"] = recon_combined
            outputs["norms"] = norms
            outputs["norms_centered"] = norms_centered
        
            if inplace: 
                self.pure_slots = pure_slots
                self.slots_ori = slots
                self.slots = slots
                self.centered_slots_ori = centered_slots
                self.centered_slots = centered_slots
                self.attns_ori = attns
                self.attns = attns
                self.normed_attns_ori = normed_attns
                self.normed_attns = normed_attns
                self.obj_pos_ori = obj_pos
                self.obj_pos = obj_pos
                self.obj_pos_filtered_ori = obj_pos_filtered
                self.obj_pos_filtered = obj_pos_filtered
                self.valid_slots_mask_ori = valid_slots_mask
                self.valid_slots_mask = valid_slots_mask
                self.background_mask_ori = background_mask
                self.background_mask = background_mask
                self.obj_exist_mask_ori = obj_exist_mask
                self.obj_exist_mask = obj_exist_mask
                # self.valid_slot_idxs_ori = None # TODO 
                # self.valid_slot_idxs = None # TODO 
                self.num_valid_slots_ori = num_valid_slots
                self.num_valid_slots = num_valid_slots
                self.recons_ori = recons
                self.recons = recons 
                self.masks_ori = masks
                self.masks = masks 
                self.img_input = img
                self.recon_combined_ori = recon_combined
                self.recon_combined = recon_combined 

                self.dists_ori = torch.zeros(1, self.num_slots)         # (1, K)
                self.dists_prev = torch.zeros(1, self.num_slots)         # (1, K)
                self.dists_centered_ori = torch.zeros(1, self.num_slots)    # (1, K)
                self.dists_centered_prev = torch.zeros(1, self.num_slots)    # (1, K)

                self.norms_ori = norms
                self.norms = norms
                self.norms_centered_ori = norms_centered
                self.norms_centered = norms_centered
        
        elif self.model_name == "slota":
            with torch.no_grad():
                img = img.to(self.device)
                outputs = self.model(img)
                recons, masks, recon_combined = self.scene_generation(outputs["slots"], visualize=visualize)
                
                attns, normed_attns = masks_to_attns(outputs["masks"])
                obj_pos = self.get_obj_pos(normed_attns) # get obj pos from alpha mask 
                obj_pos_filtered, valid_slots_mask  = self.filter_no_obj(obj_pos, attns, replace_value=-2.0) # -2 for no_obj
                obj_pos_filtered, background_mask  = self.filter_background(obj_pos_filtered, attns, replace_value=2.0) # 2 for background
                obj_exist_mask = valid_slots_mask - background_mask # (1, N_slots) # TODO: sanity check whether there is no case of valid_slots_mask not including background_mask
                num_valid_slots = torch.sum(valid_slots_mask).item()
                centered_slots = torch.zeros_like(outputs["slots"])

                norms = self.get_slot_norm(slots=outputs["slots"])   # (1, K)
                norms_centered = self.get_slot_norm(slots=centered_slots)

            outputs["attns"] = attns
            outputs["normed_attns"] = normed_attns
            outputs["obj_pos"] = obj_pos 
            outputs["obj_pos_filtered"] = obj_pos_filtered 
            outputs["valid_slots_mask"] = valid_slots_mask 
            outputs["background_mask"] = background_mask
            outputs["obj_exist_mask"] = obj_exist_mask
            # outputs["valid_slot_idxs"] = None # TODO
            outputs["num_valid_slots"] = num_valid_slots
            outputs["norms"] = norms
            outputs["norms_centered"] = norms_centered

            if inplace: 
                self.pure_slots = outputs["slots"]
                self.slots_ori = outputs["slots"]
                self.slots = outputs["slots"]
                self.centered_slots_ori = centered_slots
                self.centered_slots = centered_slots
                self.attns_ori = attns
                self.attns = attns
                self.normed_attns_ori = normed_attns
                self.normed_attns = normed_attns
                self.obj_pos_ori = obj_pos
                self.obj_pos = obj_pos
                self.obj_pos_filtered_ori = obj_pos_filtered
                self.obj_pos_filtered = obj_pos_filtered
                self.valid_slots_mask_ori = valid_slots_mask
                self.valid_slots_mask = valid_slots_mask
                self.background_mask_ori = background_mask
                self.background_mask = background_mask
                self.obj_exist_mask_ori = obj_exist_mask
                self.obj_exist_mask = obj_exist_mask
                # self.valid_slot_idxs_ori = None # TODO 
                # self.valid_slot_idxs = None # TODO 
                self.num_valid_slots_ori = num_valid_slots
                self.num_valid_slots = num_valid_slots
                self.recons_ori = outputs["recons"]
                self.recons = outputs["recons"] 
                self.masks_ori = outputs["masks"]
                self.masks = outputs["masks"] 
                self.img_input = img
                self.recon_combined_ori = outputs["recon_combined"]
                self.recon_combined = outputs["recon_combined"] 
                
                self.dists_ori = torch.zeros(1, self.num_slots)         # (1, K)
                self.dists_prev = torch.zeros(1, self.num_slots)         # (1, K)
                self.dists_centered_ori = torch.zeros(1, self.num_slots)    # (1, K)
                self.dists_centered_prev = torch.zeros(1, self.num_slots)    # (1, K)

                self.norms_ori = norms
                self.norms = norms
                self.norms_centered_ori = norms_centered
                self.norms_centered = norms_centered

        return outputs
        
    def scene_generation(
        self,
        slots: torch.tensor = torch.tensor([]), 
        visualize: str = "recons", # ["", "recon_combined", "recons"]
    ):
        if slots.nelement() == 0:
            slots = self.slots 
        if hasattr(self.model, "decoder_cnn"):
            x = self.model.decoder_cnn(slots)
        else:
            x = self.model.decoder(slots)
        print("x.shape:", x.shape)
        if x.shape[-1] == 4:
            recons, masks = x.reshape(1, self.num_slots, self.img_size, self.img_size, 4).split([3, 1], dim=-1)
        else: 
            recons = x.reshape(1, self.num_slots, self.img_size, self.img_size, 3)
            masks = torch.ones(1, self.num_slots, self.img_size, self.img_size, 1)
        # `recons`: (B, K, H, W, 3)
        # `masks`: (B, K, H, W, 1)

        masks = nn.Softmax(dim=1)(masks) # normalize alpha masks over slots
        recon_combined = torch.sum(recons * masks, dim=1)  # recombine slot-level images with alpha masks
        recon_combined = recon_combined.permute(0, 3, 1, 2)
        # `recon_combined`: (B, 3, H, W)

        if visualize == "recons": 
            self.visualize_recons(recons, masks, recon_combined)
        elif visualize == "recon_combined":
            self.visualize(recon_combined)
            
        return recons, masks, recon_combined

    # TODO: modify to take inputs
    def visualize(
        self,
        img=None, 
    ):
        if img is None:
            img = self.recon_combined
        grid = make_grid(img, nrow=1, pad_value=0).detach()
        show(grid)
    
    def visualize_recons(
        self,
        recons: torch.tensor = torch.tensor([]), 
        masks: torch.tensor = torch.tensor([]), 
        recon_combined: torch.tensor = torch.tensor([]), 
    ):
        recons_vis = recons.permute(0, 4, 2, 1, 3).reshape(1, 3, 128, -1)
        recons_vis = torch.cat([recon_combined, recons_vis], dim=-1)
        masks_vis = masks.permute(0, 4, 2, 1, 3).reshape(1, 1, 128, -1)
        masks_vis = torch.cat([torch.zeros(1, 1, 128, 128).to(self.device), masks_vis], dim=-1).repeat(1, 3, 1, 1)
        grid = torch.cat([recons_vis, masks_vis], dim=-2)
        self.visualize(grid)

    def visualize_img_input(
        self, 
    ): 
        self.visualize(self.img_input)

    def visualize_img_ori(
        self, 
    ): 
        self.visualize(self.recon_combined_ori)
    
    """ TODO: RETRIEVING """
    
    def retrieve_img(
        self, 
    ):
        pass 

    def retrieve_slot(
        self, 
    ):
        pass 

    def set_dist_metric(
        self, 
        metric: str = "mse" # ["mse", "rmse", "cosine"]    
    ):
        self.dist_metric = metric
        
    def get_closest_slot(
        self, 
    ):
        pass

    def compute_dists(
        self,
        target_slot: torch.tensor = torch.tensor([]),
        slots: torch.tensor = torch.tensor([]),
    ):
        if target_slot.nelement() == 0:
            target_slot = self.target_slot # 1, 1, slot_D

        B, K, D = slots.shape

        if self.dist_metric == "rmse": 
            return torch.sqrt(torch.sum(F.mse_loss(target_slot.repeat(B,K,1), slots, reduction="none"), dim=2))
        elif self.dist_metric == "mse": 
            return torch.sum(F.mse_loss(target_slot.repeat(B,K,1), slots, reduction="none"), dim=2)
        elif self.dist_metric == "cosine":
            return F.cosine_similarity(target_slot.repeat(B,K,1), slots, dim=2)
    
    def set_target_slot(
        self,
        target_slot: torch.tensor = torch.tensor([]),
    ):
        # target for object retrieval
        self.target_slot = target_slot
        pass 
    
    """ SLOT INVESTIGATION """
    def get_slot_dist(
        self,
        slot_idxs_src: torch.tensor = torch.tensor([]),
        slots_src: torch.tensor = torch.tensor([]),
        obj_pos_src: torch.tensor = torch.tensor([]),
        slot_idxs_tgt: torch.tensor = torch.tensor([]),
        slots_tgt: torch.tensor = torch.tensor([]),
        obj_pos_tgt: torch.tensor = torch.tensor([]),
        metric: str = "", # ["mse", "rmse", "cosine"]
        reduction: bool = False, 
    ):
        if slot_idxs_src.nelement() == 0:
            slot_idxs_src = torch.arange(self.num_slots)
            slot_idxs_tgt = torch.arange(self.num_slots)
        elif slot_idxs_tgt.nelement() == 0:
            slot_idxs_tgt = slot_idxs_src

        assert len(slot_idxs_src) == len(slot_idxs_tgt), "Indicies for slots_src and slots_tgt must be the same."

        if slots_src.nelement() == 0:
            slots_src = self.slots_ori
            obj_pos_src = self.obj_pos_filtered_ori
            
        if slots_tgt.nelement() == 0:
            slots_tgt = self.slots
            obj_pos_tgt = self.obj_pos_filtered

        if metric == "": 
            metric = self.dist_metric

        # slotaug_utils.get_slot_dist
        return get_slot_dist(slots_src[:, slot_idxs_src, ...],
                             obj_pos_src[:, slot_idxs_src, ...], 
                             slots_tgt[:, slot_idxs_tgt, ...], 
                             obj_pos_tgt[:, slot_idxs_tgt, ...], 
                             metric=metric, 
                             reduction=reduction)  

        # reduction = True  -> (, )
        # reduction = False -> (1, # slot_idxs)

    def get_slot_norm(
        self, 
        slot_idxs: torch.tensor = torch.tensor([]),
        slots: torch.tensor = torch.tensor([]),
        reduction: bool = False, 
    ):
        if slots.nelement() == 0:
            slots = self.slots 
        
        if slot_idxs.nelement() == 0:
            slot_idxs = torch.arange(self.num_slots)
        
        if reduction:
            return torch.mean(torch.norm(slots[:, slot_idxs, ...], dim=-1)) # (1, )
        else:
            return torch.norm(slots[:, slot_idxs, ...], dim=-1) # (B, K)
    
    """ DURABILITY TEST """
    def durability_test(
        self,
        num_repetitions: int = 10, 
        insts: torch.tensor = torch.tensor([]),
        slot_idxs: torch.tensor = torch.tensor([]), 
        individual: bool = False, 
        metric: str = "", # ["mse", "rmse", "cosine"]
        visualize: str = "recon_combined", 
        use_pure_slots: bool = False, 
        verbose: bool = True, 
    ):
        if metric == "": 
            metric = self.dist_metric

        num_steps = insts.shape[0]
        num_targets = len(slot_idxs)
        inv_insts = self.get_inv_insts(insts)

        outputs = {
            "slots": self.slots if not use_pure_slots else self.pure_slots,
            "centered_slots": self.centered_slots, 
            "obj_pos_filtered": self.obj_pos_filtered,
        }
        
        if verbose: 
            print("******** DURABILITY TEST ********")
            if use_pure_slots:
                print("!!! You are using PURE SLOTS !!!")
            print(f"Num Repetitions: {num_repetitions}")
            print(f"Num Steps: {num_steps}")
            print(f"Target Indicies: {slot_idxs.tolist()}")
            print("*********************************\n")
            print("--------------------------------")

        self.dura_mse.append(list())
        self.dura_rmse.append(list())
        self.dura_cosine.append(list())
        self.dura_pos.append(list())

        for i in range(num_repetitions):

            if verbose:
                print(f"Repetition: {i+1} out of {num_repetitions}")
                print()
            
            if visualize:
                recons_start, masks_start, recon_combined_start = self.scene_generation(outputs["slots"], visualize="")
                recon_combined_go_list = [recon_combined_start]
            for j in range(num_steps):
                insts[:, :num_targets, :2] = outputs["obj_pos_filtered"][0, slot_idxs]
                
                if individual:
                    for k in range(num_targets):
                        outputs = self.apply_inst_individual(
                            slot_idx=slot_idxs[k], 
                            inst=insts[j:j+1, k:k+1, ...], 
                            slots=outputs["slots"], 
                            centered_slots=outputs["centered_slots"],
                            obj_pos=outputs["obj_pos_filtered"], 
                            visualize=""
                        )
                else:
                    outputs = self.apply_insts_batch(
                        insts=insts[j:j+1], 
                        slots=outputs["slots"], 
                        centered_slots=outputs["centered_slots"],
                        obj_pos=outputs["obj_pos_filtered"], 
                        visualize=""
                    )

                slots_go = outputs["slots"]
                obj_pos_filtered_go = outputs["obj_pos_filtered"]
                if visualize:
                    recons_go, masks_go, recon_combined_go = self.scene_generation(slots_go, visualize="")
                    recon_combined_go_list.append(recon_combined_go)

                dists_ori_go = outputs["dists_ori"]
                dists_centered_ori_go = outputs["dists_centered_ori"]
                norms_go = outputs["norms"]
                norms_centered_go = outputs["norms_centered"]

            if visualize:
                if len(recon_combined_go_list) > 2:
                    recon_combined_go_list = torch.cat(recon_combined_go_list, dim=3)
                    self.visualize(recon_combined_go_list)
            
            if visualize:
                recons_start, masks_start, recon_combined_start = self.scene_generation(outputs["slots"], visualize="")
                recon_combined_back_list = [recon_combined_start]
            for j in range(num_steps):
                inv_insts[:, :num_targets, :2] = outputs["obj_pos_filtered"][0, slot_idxs]

                if individual:
                    for k in range(num_targets):
                        outputs = self.apply_inst_individual(
                            slot_idx=slot_idxs[k], 
                            inst=inv_insts[num_steps-j-1:num_steps-j, k:k+1, ...],
                            slots=outputs["slots"], 
                            centered_slots=outputs["centered_slots"],
                            obj_pos=outputs["obj_pos_filtered"], 
                            visualize=""
                        )
                else:
                    outputs = self.apply_insts_batch(
                        insts=inv_insts[num_steps-j-1:num_steps-j], 
                        slots=outputs["slots"], 
                        centered_slots=outputs["centered_slots"],
                        obj_pos=outputs["obj_pos_filtered"],
                        visualize=""
                    )
                    
                slots_back = outputs["slots"]
                obj_pos_filtered_back = outputs["obj_pos_filtered"]
                if visualize:
                    recons_back, masks_back, recon_combined_back = self.scene_generation(slots_back, visualize="")
                    recon_combined_back_list.append(recon_combined_back)

                dists_ori_back = outputs["dists_ori"]
                dists_ori_rmse_back = outputs["dists_ori_rmse"]
                dists_ori_cosine_back = outputs["dists_ori_cosine"]
                dists_centered_ori_back = outputs["dists_centered_ori"]
                norms_back = outputs["norms"]
                norms_centered_back = outputs["norms_centered"]

        
            # try:
            #     # TODO: individual obj
            #     d = torch.sqrt(torch.sum((self.obj_pos_filtered[0, slot_idxs] - obj_pos_filtered_back[0, slot_idxs])**2)).mean().item()
            #     # if d > 1:
            #     #     d = self.dura_pos[-1][-1]
            #     self.dura_mse[-1].append(dists_ori_back[0, slot_idxs].mean().item())
            #     self.dura_rmse[-1].append(dists_ori_rmse_back[0, slot_idxs].mean().item())
            #     self.dura_cosine[-1].append(dists_ori_cosine_back[0, slot_idxs].mean().item())
            #     self.dura_pos[-1].append(d)
            # except:
            d = torch.sqrt(torch.sum((self.obj_pos_filtered[0, slot_idxs[0]] - obj_pos_filtered_back[0, slot_idxs[0]])**2)).item()
            if d > 1.5:
                d = self.dura_pos[-1][-1] if len(self.dura_pos[-1]) > 0 else 1.5
            self.dura_mse[-1].append(dists_ori_back[0, slot_idxs[0]].item())
            self.dura_rmse[-1].append(dists_ori_rmse_back[0, slot_idxs[0]].item())
            self.dura_cosine[-1].append(dists_ori_cosine_back[0, slot_idxs[0]].item())
            self.dura_pos[-1].append(d)
            
            if visualize:
                if len(recon_combined_back_list) > 2:
                    recon_combined_back_list = torch.cat(recon_combined_back_list, dim=3)
                    self.visualize(recon_combined_back_list)

            if visualize == "recons":
                recons_go = recons_go.permute(0, 4, 2, 1, 3).reshape(1, 3, 128, -1)
                recons_go = torch.cat([recon_combined_go, recons_go], dim=3)
                recons_back = recons_back.permute(0, 4, 2, 1, 3).reshape(1, 3, 128, -1)
                recons_back = torch.cat([recon_combined_back, recons_back], dim=3)
                self.visualize(
                    torch.cat([recons_go, recons_back], dim=-2)
                )
            elif visualize == "recon_combined":
                self.visualize(
                    torch.cat([recon_combined_go, recon_combined_back], dim=3)
                )

            for slot_idx in slot_idxs: 
                slot_idx = slot_idx.unsqueeze(0)
                
                if verbose:
                    print("slot_idx:{}".format(slot_idx.item()))

                    print("norm (go): {:.4f}".format(norms_go[0, slot_idx].item()))
                    print("norm_ori_dist (go): {:.4f}".format(
                        torch.sqrt((norms_go[0, slot_idx] - self.norms_ori[0, slot_idx]) ** 2).item()
                    ))
                    # print("norm_prev_dist (go): {:.4f}".format(
                    #     torch.sqrt((norms_go[0, slot_idx] - norms_prev_go[0, slot_idx]) ** 2).item()
                    # ))
                    print("norm_centered_ori_dist (go): {:.4f}".format(
                        torch.sqrt((norms_centered_go[0, slot_idx] - self.norms_centered_ori[0, slot_idx]) ** 2).item()
                    ))
                    # print("norm_centered_prev_dist (go): {:.4f}".format(
                    #     torch.sqrt((norms_centered_go[0, slot_idx] - norms_centered_prev_go[0, slot_idx]) ** 2).item()
                    # ))

                    print("norm (back): {:.4f}".format(norms_back[0, slot_idx].item()))
                    print("norm_ori_dist (back): {:.4f}".format(
                        torch.sqrt((norms_back[0, slot_idx] - self.norms_ori[0, slot_idx]) ** 2).item()
                    ))
                    # print("norm_prev_dist (back): {:.4f}".format(
                    #     torch.sqrt((norms_back[0, slot_idx] - norms_prev_back[0, slot_idx]) ** 2).item()
                    # ))
                    print("norm_centered_ori_dist (back): {:.4f}".format(
                        torch.sqrt((norms_centered_back[0, slot_idx] - self.norms_centered_ori[0, slot_idx]) ** 2).item()
                    ))
                    # print("norm_centered_prev_dist (back): {:.4f}".format(
                    #     torch.sqrt((norms_centered_back[0, slot_idx] - norms_centered_prev_back[0, slot_idx]) ** 2).item()
                    # ))

                    print("dist_ori (go): {:.4f}".format(dists_ori_go[0, slot_idx].item()))
                    # print("dist_prev (go): {:.4f}".format(dists_prev_go[0, slot_idx].item()))
                    print("dist_centered_ori (go): {:.4f}".format(dists_centered_ori_go[0, slot_idx].item()))
                    # print("dist_centered_prev (go): {:.4f}".format(dists_centered_prev_go[0, slot_idx].item()))

                    print("dist_ori (back): {:.4f}".format(dists_ori_back[0, slot_idx].item()))
                    # print("dist_prev (back): {:.4f}".format(dists_prev_back[0, slot_idx]))
                    print("dist_centered_ori (back): {:.4f}".format(dists_centered_ori_back[0, slot_idx].item()))
                    # print("dist_centered_prev (back): {:.4f}".format(dists_centered_prev_back[0, slot_idx].item()))

                    print()
            if verbose:
                print("--------------------------------")
