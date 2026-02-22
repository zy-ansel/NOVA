import torch
from typing import Tuple, Callable
import torch
import math
from typing import Type, Dict, Any, Tuple, Callable
from torch.nn import functional as F

def compute_layer_entropy(
    cur_x: torch.Tensor,
    H: int,
    W: int,
    mode: str = "channel",
    eps: float = 1e-6,
    normalize: bool = True,
    dtype_float: torch.dtype = torch.float32,
):
    B, L, C = cur_x.shape
    x = cur_x.view(B, H, W, C).permute(0, 3, 1, 2)
    if mode == "channel":
        feat = cur_x.to(dtype_float)
        energy_c = (feat ** 2).clamp_min(eps)  # (B, L, C)
        denom = energy_c.sum(dim=-1, keepdim=True).clamp_min(eps)
        p = energy_c / denom                    # (B, L, C)

        ent = -(p * (p + eps).log()).sum(dim=-1, keepdim=True)  # (B, L, 1)
        if normalize:
            ent = ent / math.log(C + 1e-12)

    ent_mean = ent.mean(dim=1)
    ent_var = ent.var(dim=1, correction=0)
    ent_std = ent_var.sqrt()

    ent_values = ent.view(-1)
    mean_val = ent_values.mean()
    median_val = ent_values.median()
    diff = mean_val - median_val
    lower, upper = min(mean_val, median_val), max(mean_val, median_val)
    mask = (ent_values >= lower) & (ent_values <= upper)
    count = mask.sum().float()

    if diff < 0:
        ratio = (ent_values.numel() + count // 2) / ent_values.numel()
    else :
        ratio = (ent_values.numel() - count // 2) / ent_values.numel()
    
    return {
        "ent": ent,
        "ent_mean": ent_mean,
        "ent_var": ent_var,
        "ent_std": ent_std,
        "ratio": ratio,
        "diff": diff,
    }

def do_nothing(x: torch.Tensor, *args, **kwargs):
    return x


def masked_previous_scale_cache(cur_x, num_remain, cur_shape):
    B, L, c = cur_x.shape
    # mean_x = cur_x.view(B, cur_shape[1], cur_shape[2], -1).permute(0, 3, 1, 2)
    # mean_x = torch.nn.functional.adaptive_avg_pool2d(mean_x,(1,1)).permute(0, 2, 3, 1).view(B, 1,c)
    # mse_difference = torch.sum((cur_x - mean_x)**2,dim=-1,keepdim=True)
    # select_indices = torch.argsort(mse_difference,dim=1,descending=True)
    # filted_select_indices=select_indices[:,:num_remain,:]
    H, W = cur_shape[1], cur_shape[2]
    entropy_results = compute_layer_entropy(cur_x, H, W)

    # 提取熵图，形状为 (B, L, 1)
    importance_score = entropy_results["ent"] 
    
    # --- 2. 根据重要性排序 ---
    # 熵越大，代表信息量越丰富，越需要保留，所以使用 descending=True
    select_indices = torch.argsort(importance_score, dim=1, descending=True)
    
    # 获取要保留的 Token 索引
    filted_select_indices = select_indices[:, :num_remain, :]


    def merge(merged_cur_x):
        return torch.gather(merged_cur_x,dim=1,index=filted_select_indices.repeat(1,1,c))

    def unmerge(unmerged_cur_x, unmerged_cache_x, cached_hw=None):
        unmerged_cache_x_ = unmerged_cache_x.view(B, cached_hw[0], cached_hw[1], -1).permute(0, 3, 1, 2)
        unmerged_cache_x_ = torch.nn.functional.interpolate(unmerged_cache_x_, size=(cur_shape[1], cur_shape[2]), mode='area').permute(0, 2, 3, 1).view(B, L, c)
        unmerged_cache_x_.scatter_(dim=1,index=filted_select_indices.repeat(1,1,c),src=unmerged_cur_x)
        return unmerged_cache_x_

    def get_src_tgt_idx():
        return filted_select_indices

    return merge, unmerge, get_src_tgt_idx

# def compute_layer_wise_ratio(entropy_mean, cur_entropy_mean, ratio):


def masked_previous_scale_cache_dynamic(cur_x, num_remain, cur_shape, pre_mean):
    B, L, C = cur_x.shape
    # b, ph, pw = cur_scale

    ent_out = compute_layer_entropy(
        cur_x=cur_x,
        H=cur_shape[1],
        W=cur_shape[2]
    )

    ent_local = ent_out["ent"]
    ratio = ent_out["ratio"]
    # print(ratio)
    ent_mean = ent_out["ent_mean"]
    mean_ratio = (ent_mean - pre_mean) / (ent_mean + pre_mean)
    new_ratio = mean_ratio.mean().item() + 1
    new_ent = 0.5 * (ent_mean + pre_mean)
    num_remain = math.ceil(num_remain * new_ratio)
    num_remain = max(0, min(num_remain, L))
    score = ent_local.squeeze(-1)
    cur_ratio = num_remain / L

    topk_idx = torch.topk(score, k=num_remain, dim=1, largest=True, sorted=False).indices
    topk_idx = topk_idx.unsqueeze(-1)

    def merge(x):
        # 用 expand 而不是 repeat，避免复制索引
        idx = topk_idx.expand(-1, -1, C)    # (B, num_remain, C)
        return torch.gather(x, dim=1, index=idx)

    def unmerge(x_small, x_cache, cached_hw=None):
        Hc, Wc = cached_hw
        x_cache_ = x_cache.view(B, Hc, Wc, C).permute(0, 3, 1, 2)
        x_cache_ = F.interpolate(x_cache_, size=(cur_shape[1], cur_shape[2]), mode='area').permute(0, 2, 3, 1).reshape(B, L, C)
        out = x_cache_.clone()
        idx = topk_idx.expand(-1, -1, C)
        out.scatter_(dim=1, index=idx, src=x_small)
        return out

    def get_src_tgt_idx():
        return topk_idx

    return merge, unmerge, get_src_tgt_idx, new_ent


# 1/2 : [... (1, 23, 46), (1, 30, 60), (1, 37, 74), (1, 45, 90), (1, 60, 120)]
# 1.333/1  (1, 36, 27), (1, 48, 36), (1, 60, 45), (1, 72, 54) (1,84,63)
# 2/1:  (1, 46, 23), (1, 60, 30), (1, 74, 37), (1, 90, 45) (1,120,60)
# 1/1 , (13, 32, 32), (15, 40, 40), (17, 48, 48), (21, 64, 64), (1, 84, 84)]
def compute_merge(
        x: torch.Tensor, 
        prune_scale_list=[20, 24, 32, 40, 48, 60], 
        is_later_layer=False, 
        x_shape=None,
        entropy=None,
        cur_scale=None,
        puring_ratio=0.0,
        previous_e_mean=0.0,
        rep_pick: str = "first",  
        weight_mode: str = "mean",      
        keep_M_equal: bool = True,       
        fuse_prev: float = 1.0,          
    ) -> Tuple[Callable, ...]:
    _, original_h, original_w = x_shape
    original_tokens = original_h * original_w

    if original_w in prune_scale_list and is_later_layer:
        ratio_hard_code = {20: 0.10, 24: 0.15, 32:0.4, 40:0.5, 48: 0.90, 60: 0.95}
        # ratio =ratio_hard_code[original_w]
        ratio=puring_ratio
        r = int(x.shape[1] * ratio)
        m, u, id_fn, new_e_mean = masked_previous_scale_cache_dynamic(x,x.shape[1]-r,x_shape, previous_e_mean)
    else:
        m, u, id_fn = (do_nothing, do_nothing, do_nothing)
        new_e_mean = 0.0

    m_a, u_a = (m, u)

    return m_a, u_a, id_fn, new_e_mean  # Okay this is probably not very good


