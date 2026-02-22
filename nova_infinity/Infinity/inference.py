import random
import os
import torch
import cv2
import numpy as np
from tools.run_infinity import *
from contextlib import contextmanager
import gc
import argparse
import os.path as osp

# =========================
# Anonymized paths for double-blind review
# =========================
ROOT_DIR = "/path/to/project_root"

model_path = osp.join(ROOT_DIR, "checkpoints", "infinity_2b.pth")
vae_path   = osp.join(ROOT_DIR, "checkpoints", "infinity_vae_d32.pth")
text_encoder_ckpt = osp.join(ROOT_DIR, "checkpoints", "t5xl")

args = argparse.Namespace(
    pn='1M',  # 1M, 0.60M, 0.25M, 0.06M
    model_path=model_path,
    cfg_insertion_layer=0,
    vae_type=32,
    vae_path=vae_path,
    add_lvl_embeding_only_first_block=1,
    use_bit_label=1,
    model_type='infinity_2b',
    rope2d_each_sa_layer=1,
    rope2d_normalized_by_hw=2,
    use_scale_schedule_embedding=0,
    sampling_per_bits=1,
    text_encoder_ckpt=text_encoder_ckpt,
    text_channels=2048,
    apply_spatial_patchify=0,
    h_div_w_template=1.000,
    use_flex_attn=0,
    cache_dir=osp.join(ROOT_DIR, "cache"),
    checkpoint_type='torch',
    seed=0,
    bf16=1,
    save_file='tmp.jpg'
)

# load text encoder
text_tokenizer, text_encoder = load_tokenizer(
    t5_path=args.text_encoder_ckpt
)

# load vae
vae = load_visual_tokenizer(args)

# load infinity
infinity = load_transformer(vae, args)

# =====================================================
# Inference
# =====================================================
prompt = (
    "A cute and fluffy teddy bear sitting on a wooden shelf, "
    "its fur a soft caramel color, with button eyes and a stitched smile. "
    "The room is warmly lit, surrounded by colorful children's books and toys."
)

cfg = 4
tau = 0.5
h_div_w = 1.0
seed = random.randint(0, 10000)
enable_positive_prompt = 0

h_div_w_template_ = h_div_w_templates[
    np.argmin(np.abs(h_div_w_templates - h_div_w))
]

scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)


# =========================
# Memory measurement
# =========================
@contextmanager
def measure_peak_memory(device=None):
    device = torch.cuda.current_device() if device is None else device
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    yield
    torch.cuda.synchronize(device)
    peak_alloc = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    peak_resv  = torch.cuda.max_memory_reserved(device) / 1024 / 1024
    print(
        f"[PyTorch] peak allocated={peak_alloc:.2f} MB | "
        f"reserved={peak_resv:.2f} MB"
    )


with torch.inference_mode():
    with measure_peak_memory():
        for _ in range(10):
            start_event.record()
            generated_image = gen_one_img(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                prompt,
                g_seed=seed,
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=cfg,
                tau_list=tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=enable_positive_prompt,
            )

args.save_file = "output.jpg"
os.makedirs(osp.dirname(osp.abspath(args.save_file)), exist_ok=True)
cv2.imwrite(args.save_file, generated_image.cpu().numpy())
print(f"Saved to {osp.abspath(args.save_file)}")