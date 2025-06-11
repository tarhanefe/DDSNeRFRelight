from typing import Union, Optional, List, Tuple, Dict, Callable, Any
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image
from diffusers import DDIMScheduler
from torchvision import transforms as tfms
from diffusers.models.attention_processor import Attention
from dataclasses import dataclass, field
import math
from diffusers import StableDiffusionPipeline
from pytorch_wavelets import DWTForward, DWTInverse


###############################################################
class MyCrossAttnProcessor:
    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        query = attn.to_q(hidden_states)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # save text-conditioned attention map only
        # get attention map of ref
        if hidden_states.shape[0] == 4: 
            attn.hs = hidden_states[2:3]
        # get attention map of trg
        else:
            attn.hs = hidden_states[1:2]

        return hidden_states

def prep_unet(unet):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention":
            module.set_processor(MyCrossAttnProcessor())
    return unet

###############################################################

# PatchNCE loss from https://github.com/taesungp/contrastive-unpaired-translation
# https://github.com/YSerin/ZeCon/blob/main/optimization/losses.py

class CutLoss:
    def __init__(self, n_patches=256, patch_size=1):
        self.n_patches = n_patches
        self.patch_size = patch_size
    
    def get_attn_cut_loss(self, ref_noise, trg_noise):
        loss = 0

        bs, res2, c = ref_noise.shape
        res = int(np.sqrt(res2))
        sh1 = res
        sh2 = res

        ref_noise_reshape = ref_noise.reshape(bs, sh1, sh2, c).permute(0, 3, 1, 2) 
        trg_noise_reshape = trg_noise.reshape(bs, sh1, sh2, c).permute(0, 3, 1, 2)

        for ps in self.patch_size:
            if ps > 1:
                pooling = nn.AvgPool2d(kernel_size=(ps, ps))
                ref_noise_pooled = pooling(ref_noise_reshape)
                trg_noise_pooled = pooling(trg_noise_reshape)
            else:
                ref_noise_pooled = ref_noise_reshape
                trg_noise_pooled = trg_noise_reshape

            ref_noise_pooled = nn.functional.normalize(ref_noise_pooled, dim=1)
            trg_noise_pooled = nn.functional.normalize(trg_noise_pooled, dim=1)

            ref_noise_pooled = ref_noise_pooled.permute(0, 2, 3, 1).flatten(1, 2)
            patch_ids = np.random.permutation(ref_noise_pooled.shape[1]) 
            patch_ids = patch_ids[:int(min(self.n_patches, ref_noise_pooled.shape[1]))]
            patch_ids = torch.tensor(patch_ids, dtype=torch.long, device=ref_noise.device)

            ref_sample = ref_noise_pooled[:1, patch_ids, :].flatten(0, 1)

            trg_noise_pooled = trg_noise_pooled.permute(0, 2, 3, 1).flatten(1, 2) 
            trg_sample = trg_noise_pooled[:1 , patch_ids, :].flatten(0, 1) 
            
            loss += self.PatchNCELoss(ref_sample, trg_sample).mean() 
        return loss

    def PatchNCELoss(self, ref_noise, trg_noise, batch_size=1, nce_T = 0.07):
        batch_size = batch_size
        nce_T = nce_T
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        mask_dtype = torch.bool

        num_patches = ref_noise.shape[0]
        dim = ref_noise.shape[1]
        ref_noise = ref_noise.detach()
        
        l_pos = torch.bmm(
            ref_noise.view(num_patches, 1, -1), trg_noise.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1) 

        # reshape features to batch size
        ref_noise = ref_noise.view(batch_size, -1, dim)
        trg_noise = trg_noise.view(batch_size, -1, dim) 
        npatches = ref_noise.shape[1]
        l_neg_curbatch = torch.bmm(ref_noise, trg_noise.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=ref_noise.device, dtype=mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0) 
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / nce_T

        loss = cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=ref_noise.device))

        return loss

class DDSLoss:

    def noise_input(self, z, eps=None, timestep: Optional[int]= None):
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(
                low = self.t_min,
                high = min(self.t_max, 1000) -1,
                size=(b,),
                device=z.device,
                dtype=torch.long
            )

        if eps is None:
            eps = torch.randn_like(z)

        z_t = self.scheduler.add_noise(z, eps, timestep)
        return z_t, eps, timestep
    
    def get_epsilon_prediction(self, z_t, timestep, embedd, guidance_scale=7.5, cross_attention_kwargs=None):

        latent_input = torch.cat([z_t] * 2)
        timestep = torch.cat([timestep] * 2)
        embedd = embedd.permute(1, 0, 2, 3).reshape(-1, *embedd.shape[2:])

        e_t = self.unet(latent_input, timestep, embedd, cross_attention_kwargs=cross_attention_kwargs,).sample
        e_t_uncond, e_t = e_t.chunk(2)
        e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
        assert torch.isfinite(e_t).all()

        return e_t

    def __init__(self, t_min, t_max, unet, scheduler, device):
        self.t_min = t_min
        self.t_max = t_max
        self.unet = unet
        self.scheduler = scheduler
        self.device = device


###############################################################

@dataclass
class DCConfig:
    sd_pretrained_model_or_path: str = "CompVis/stable-diffusion-v1-4"
    min_step_ratio: float = 0.2
    max_step_ratio: float = 0.9
    num_inference_steps: int = 1000
    src_prompt: str = "a photo of a sks man"
    tgt_prompt: str = "a photo of a Batman"
    log_step: int = 10
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5
    device: torch.device = torch.device("cuda")
    wavelet_filtering: bool = True
    wavelet_name: str = "db8"
    wavelet_level: int = 3
    n_patches: int = 256
    patch_size: list = (1, 2)
    w_dds: float = 1.0
    w_cut: float = 3.0
    scheduler_pretrained_path: Optional[str] = None
    loss_multiplier: float = 0.02
    psi: float=0.075
    chi = math.log(0.1)
    delta: float=0.2
    gamma: float=0.8
    freeu_b1: float=1.1
    freeu_b2: float=1.1
    freeu_s1: float=0.9
    freeu_s2: float=0.2
    pipeline: str = "cds"
class DC:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.pipe = StableDiffusionPipeline.from_pretrained(config.sd_pretrained_model_or_path).to(self.device)
        self.unet = self.pipe.unet
        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.num_train_timesteps = config.num_inference_steps
        self.t_min = int(self.num_train_timesteps * 0.05)
        self.t_max = int(self.num_train_timesteps * 0.95)
        self.cut_loss = CutLoss(config.n_patches, config.patch_size)
        self.dds_loss = DDSLoss(self.t_min, self.t_max, self.unet, self.scheduler, self.device)
        self.scheduler.set_timesteps(config.num_inference_steps)

        self.unet.requires_grad_(False)
        self.unet = prep_unet(self.pipe.unet)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        self.src_prompt = self.config.src_prompt
        self.tgt_prompt = self.config.tgt_prompt

        self.update_text_features(src_prompt=self.src_prompt, tgt_prompt=self.tgt_prompt)
        self.null_text_feature = self.encode_text("")
        
    def __call__(self, tgt_x0, src_x0, src_emb, tgt_prompt=None, src_prompt=None, reduction="mean", return_dict=False, step=0, current_spot=0):
        device = self.device
        z_src = src_x0
        z_trg = tgt_x0#.detach().requires_grad_(True)
        batch_size = tgt_x0.shape[0]
    
        # Update text embeddings
        self.update_text_features(src_prompt=src_prompt, tgt_prompt=tgt_prompt)
        trg_prompt_embeds, prompt_embeds = (
            self.tgt_text_feature,
            self.src_text_feature,
        )
    
        sa_attn = {}
    
        # Get shared noise and timestep
        z_t_src, eps, timestep = self.dds_loss.noise_input(z_src)
        z_t_trg, _, _ = self.dds_loss.noise_input(z_trg, eps, timestep)
        
        # Predict epsilon
        eps_pred = self.dds_loss.get_epsilon_prediction(
            torch.cat((z_t_src, z_t_trg)),
            torch.cat((timestep, timestep)),
            torch.cat((prompt_embeds, trg_prompt_embeds))
        )
        eps_src, eps_trg = eps_pred.chunk(2)
        grad_dds = (eps_trg - eps_src)
        sa_attn[timestep.item()] = {}
        # Store reference attention maps
        for name, module in self.unet.named_modules(): 
            module_name = type(module).__name__
            
            if module_name == "Attention":
                if "attn1" in name and "up" in name:
                    hidden_state = module.hs
                    sa_attn[timestep.item()][name] = hidden_state.detach().cpu()
        # Manual DDS gradient
        
        manual_dds_grad = 2000 * self.config.w_dds * grad_dds / (z_trg.shape[2] * z_trg.shape[3])
    
        # Inject DDS gradient manually
        #z_trg.backward(gradient=manual_dds_grad, retain_graph=True)
    
        # Forward again to get trg attention maps after DDS gradient
        z_t_trg, _, _ = self.dds_loss.noise_input(z_trg, eps, timestep)
        eps_trg = self.dds_loss.get_epsilon_prediction(z_t_trg, timestep, trg_prompt_embeds)
    
        # CUT Loss
        cutloss = 0
        for name, module in self.unet.named_modules(): 
            module_name = type(module).__name__
            if module_name == "Attention":
                # sa_cut
                if "attn1" in name and "up" in name:
                    curr = module.hs
                    ref = sa_attn[timestep.item()][name].detach().to(device)
                    cutloss += self.cut_loss.get_attn_cut_loss(ref, curr)

        cut_loss_val = cutloss * self.config.w_cut
    
        cut_grad = torch.autograd.grad(
            outputs=cut_loss_val,
            inputs=z_trg,
            grad_outputs=torch.ones_like(cut_loss_val),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        # Get final combined gradient
        grad = manual_dds_grad + cut_grad 
        
        # Optional wavelet filtering
        if self.config.wavelet_filtering:
            grad = self.wave_grad(
                grad,
                wavelet=self.config.wavelet_name,
                level=self.config.wavelet_level,
            )
        target = (tgt_x0 - grad).detach() 
        loss = 0.5 * F.mse_loss(tgt_x0, target, reduction=reduction) / batch_size * self.config.loss_multiplier

        #torch.cuda.empty_cache()
        #del eps_pred, eps_src, eps_trg, z_t_src, z_t_trg
        
        if return_dict:
            return {
                "loss": loss,
                "grad": grad,
                "t": timestep,
            }
        return loss

    def encode_image(self, img_tensor):
        x = img_tensor.float()
        x = 2 * x - 1
        latents = self.vae.encode(img_tensor)
        return latents['latent_dist'].mean * 0.18215
    
    def encode_src_image(self, img_tensor):
        x = img_tensor
        x = 2 * x - 1
        latents = self.vae.encode(img_tensor)
        return latents['latent_dist'].mean * 0.18215
    
    def decode_latents(self, latents):
        x = self.vae.decode(latents / 0.18215).sample
        x = (x / 2 + 0.5).clamp(0, 1)
        return x.cpu().permute(0, 2, 3, 1).numpy()

    def encode_text(self, prompt: Union[str, List[str]], negative_prompt: Optional[str] = ""):
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        
        text_inputs = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt").to(self.device)
        prompt_embeds = self.text_encoder(text_inputs.input_ids)[0]
    
        negative_inputs = self.tokenizer([negative_prompt] * batch_size, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt").to(self.device)
        negative_embeds = self.text_encoder(negative_inputs.input_ids)[0]
    
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        negative_embeds = negative_embeds.to(dtype=self.text_encoder.dtype)
        negative_embeds = negative_embeds.repeat(1, 1, 1)
        negative_embeds = negative_embeds.view(bs_embed * 1, seq_len, -1)
    
        prompt_embeds = torch.stack([negative_embeds, prompt_embeds], axis=1)
        return prompt_embeds

    def update_text_features(self, src_prompt=None, tgt_prompt=None):
        if getattr(self, "src_text_feature", None) is None:
            assert src_prompt is not None
            self.src_prompt = src_prompt
            self.src_text_feature = self.encode_text(src_prompt)
        else:
            if src_prompt is not None and src_prompt != self.src_prompt:
                self.src_prompt = src_prompt
                self.src_text_feature = self.encode_text(src_prompt)

        if getattr(self, "tgt_text_feature", None) is None:
            assert tgt_prompt is not None
            self.tgt_prompt = tgt_prompt
            self.tgt_text_feature = self.encode_text(tgt_prompt)
        else:
            if tgt_prompt is not None and tgt_prompt != self.tgt_prompt:
                self.tgt_prompt = tgt_prompt
                self.tgt_text_feature = self.encode_text(tgt_prompt)
        
    def wave_grad(self, grad: torch.Tensor, wavelet: str = 'db2', level: int = 1) -> torch.Tensor:
        """
        Wavelet-based low-pass filtering using pytorch_wavelets.
        Args:
            grad: Tensor of shape [B, C, H, W]
            wavelet: Wavelet type (e.g., 'db2', 'haar', 'sym2')
            level: Number of decomposition levels
        Returns:
            Low-pass filtered tensor of same shape
        """
        dwt = DWTForward(J=level, wave=wavelet, mode='zero').to(grad.device)
        idwt = DWTInverse(wave=wavelet, mode='zero').to(grad.device)
    
        Yl, Yh = dwt(grad)  # Yl: lowpass, Yh: list of highpass components
        Yh_filtered = [torch.zeros_like(h) for h in Yh]  # remove high-freq info
    
        grad_lp = idwt((Yl, Yh_filtered))
        # Ensure shape match by cropping to original size
        grad_lp = grad_lp[:, :, :grad.shape[2], :grad.shape[3]]  # Crop H, W to original
        return grad_lp

def tensor_to_pil(img):
    if img.ndim == 4:
        img = img[0]
    img = img.cpu().permute(1, 2, 0).detach().numpy()
    
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    return img


def pil_to_tensor(img, device="cpu"):
    device = torch.device(device)
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img[None].transpose(0, 3, 1, 2))
    img = img.to(device)
    return img


def resize_image(image, min_size):
    if min(image.size) < min_size:
        image = image.resize((min_size, min_size))
    return image
