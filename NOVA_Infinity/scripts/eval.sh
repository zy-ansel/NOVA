#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# ================= Configuration Section =================
# 请根据实际环境修改以下路径，避免使用包含个人ID或特定会议名的绝对路径
# 建议在提交代码时使用相对路径或通用的挂载路径

# 假设当前脚本在项目根目录下运行
PROJ_ROOT=$(pwd)

# 模型权重存放目录
CKPT_ROOT="${PROJ_ROOT}/ckpts"
# 外部数据集存放目录
DATA_ROOT="${PROJ_ROOT}/datasets"
# 结果输出目录
OUTPUT_ROOT="${PROJ_ROOT}/results"

# 设置 PYTHONPATH
export PYTHONPATH="${PROJ_ROOT}:$PYTHONPATH"

# Python 环境设置
python_ext=python3
pip_ext=pip3

# =========================================================

infer_eval_image_reward() {
    # step 1, infer images (Commented out in original)
    # ... (omitted for brevity, same as original)

    # step 2, compute image reward
    ${python_ext} evaluation/image_reward/cal_imagereward.py \
    --meta_file ${out_dir}/metadata.jsonl
}

infer_eval_hpsv21() {
    # 动态获取当前环境的 site-packages 路径，避免泄露 Anaconda 用户名路径
    SITE_PACKAGES=$(${python_ext} -c "import site; print(site.getsitepackages()[0])")
    HPSV2_CLIP_PATH="${SITE_PACKAGES}/hpsv2/src/open_clip"
    
    wget https://dl.fbaipublicfiles.com/mmf/clip/bpe_simple_vocab_16e6.txt.gz
    
    # 检查目标目录是否存在，避免报错
    if [ -d "$HPSV2_CLIP_PATH" ]; then
        mv bpe_simple_vocab_16e6.txt.gz "$HPSV2_CLIP_PATH"
    else
        echo "Warning: HPSv2 path not found at $HPSV2_CLIP_PATH"
    fi

    mkdir -p ${out_dir}
    ${python_ext} evaluation/hpsv2/eval_hpsv2.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --outdir ${out_dir}/images | tee ${out_dir}/log.txt
}

test_dpg() {
    ${python_ext} evaluation/dpg/infer4eval.py \
        --cfg ${cfg} \
        --tau ${tau} \
        --pn ${pn} \
        --model_path ${infinity_model_path} \
        --vae_type ${vae_type} \
        --vae_path ${vae_path} \
        --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
        --use_bit_label ${use_bit_label} \
        --model_type ${model_type} \
        --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
        --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
        --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
        --cfg ${cfg} \
        --tau ${tau} \
        --checkpoint_type ${checkpoint_type} \
        --text_encoder_ckpt ${text_encoder_ckpt} \
        --text_channels ${text_channels} \
        --apply_spatial_patchify ${apply_spatial_patchify} \
        --save_file tmp.jpg \
        --datasets DPG \
        -o ${out_dir}
    
    cd cus_datasets
    bash dpg_bench/dist_eval.sh ../${out_dir}/DPG 1024
    cd ..
}


test_mjhq() {
    log_dir="${out_dir}"
    mkdir -p "${log_dir}"
    {
        ${python_ext} evaluation/mjhq/infer4eval.py \
            --cfg ${cfg} \
            --tau ${tau} \
            --pn ${pn} \
            --model_path ${infinity_model_path} \
            --vae_type ${vae_type} \
            --vae_path ${vae_path} \
            --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
            --use_bit_label ${use_bit_label} \
            --model_type ${model_type} \
            --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
            --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
            --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
            --checkpoint_type ${checkpoint_type} \
            --text_encoder_ckpt ${text_encoder_ckpt} \
            --text_channels ${text_channels} \
            --apply_spatial_patchify ${apply_spatial_patchify} \
            --cfg_insertion_layer ${cfg_insertion_layer} \
            --outdir ${out_dir}/images

        # calculate fid
        ${python_ext} tools/fid_score.py \
            ${out_dir}/images/people \
            ${mjhq_gt_path}/people
    } 2>&1 | tee "${log_dir}/log.txt"
}

test_gen_eval() {
    # 路径已改为基于 PROJ_ROOT 的相对路径
    GEN_EVAL_ROOT="${PROJ_ROOT}/evaluation/gen_eval"
    
    # run inference
    ${python_ext} evaluation/gen_eval/infer4eval.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --outdir ${out_dir}/images \
    --rewrite_prompt ${rewrite_prompt} \
    --metadata_file ${GEN_EVAL_ROOT}/prompts/evaluation_metadata.jsonl

    # detect objects
    ${python_ext} ${GEN_EVAL_ROOT}/evaluate_images.py ${out_dir}/images \
    --outfile ${out_dir}/results/det.jsonl \
    --model-config ${GEN_EVAL_ROOT}/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py \
    --model-path ${CKPT_ROOT}

    # accumulate results
    ${python_ext} ${GEN_EVAL_ROOT}/summary_scores.py ${out_dir}/results/det.jsonl > ${out_dir}/results/res.txt
    cat ${out_dir}/results/res.txt
}

test_fid() {
    ${pip_ext} install pytorch_fid

    # step 1, infer images
    ${python_ext} tools/comprehensive_infer.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --coco30k_prompts 0 \
    --save4fid_eval 1 \
    --jsonl_filepath ${jsonl_filepath} \
    --long_caption_fid ${long_caption_fid} \
    --out_dir  ${out_dir} \

    # step 2, compute fid
    ${python_ext} tools/fid_score.py \
    ${out_dir}/pred \
    ${out_dir}/gt | tee ${out_dir}/log.txt
}

test_val_loss() {
    ${python_ext} evaluation/validation_loss/validation_loss.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --save_dir ${out_dir} \
    --reweight_loss_by_scale ${reweight_loss_by_scale} \
    --meta_folder ${jsonl_folder} \
    --noise_apply_strength ${noise_apply_strength} \
    --bf16 0 \
    --log_freq 10
}

# ================= Inference Arguments =================
pn=1M
model_type=infinity_2b
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_channels=2048
apply_spatial_patchify=0
cfg_insertion_layer=0

# 使用之前定义的 ROOT 变量拼接路径
infinity_model_path="${CKPT_ROOT}/infinity_2b_reg.pth"
vae_path="${CKPT_ROOT}/infinity_vae_d32reg.pth"
text_encoder_ckpt="${CKPT_ROOT}/t5xl"

# 定义输出根目录 (匿名化)
out_dir_root="${OUTPUT_ROOT}/eval_results/2b_model"

vae_type=32
cfg=4
tau=1
sub_fix=cfg${cfg}_tau${tau}_cfg_insertion_layer${cfg_insertion_layer}

# ================= Execution Blocks =================

# ImageReward
out_dir=${out_dir_root}/image_reward_${sub_fix}
# infer_eval_image_reward

# HPS v2.1
out_dir=${out_dir_root}/hpsv21_${sub_fix}
# infer_eval_hpsv21

# GenEval
rewrite_prompt=1
out_dir=${out_dir_root}/gen_eval_${sub_fix}_rewrite_prompt${rewrite_prompt}_round2_real_rewrite
# test_gen_eval

# long caption fid
long_caption_fid=1
jsonl_filepath='[YOUR VAL JSONL FILEPATH]' # 需手动填入通用路径
out_dir=${out_dir_root}/val_long_caption_fid_${sub_fix}
rm -rf ${out_dir}
# test_fid

# test val loss
out_dir=${out_dir_root}/val_loss_${sub_fix}
reweight_loss_by_scale=0
jsonl_folder='[YOUR VAL JSONL FILEPATH]' # 需手动填入通用路径
noise_apply_strength=0.2
# test_val_loss

# mjhq
out_dir=${out_dir_root}/mjhq_${sub_fix}——sta
# 使用 DATA_ROOT 替代绝对路径
mjhq_gt_path="${DATA_ROOT}/MJHQ/MJHQ-30K/image" 
test_mjhq


# dpg
out_dir=${out_dir_root}/dpg_${sub_fix}
# test_dpg