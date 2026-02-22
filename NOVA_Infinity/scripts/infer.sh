#!/bin/bash

############################################################
# Anonymous-safe runtime script
############################################################

# automatically set project root as current script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}:$PYTHONPATH"

############################################################
# user-fill paths (anonymous safe placeholders)
############################################################

MODEL_PATH="[PATH_TO_MODEL_CHECKPOINT]"
VAE_PATH="[PATH_TO_VAE_CHECKPOINT]"
TEXT_ENCODER_PATH="[PATH_TO_TEXT_ENCODER]"
OUTPUT_IMAGE="[OUTPUT_IMAGE_PATH]"

############################################################
# inference config
############################################################

pn=1M
model_type=infinity_2b
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'

infinity_model_path=${MODEL_PATH}
vae_type=32
vae_path=${VAE_PATH}

cfg=4
tau=0.5
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1

text_encoder_ckpt=${TEXT_ENCODER_PATH}
text_channels=2048
apply_spatial_patchify=0

############################################################
# run inference
############################################################

python3 tools/run_infinity.py \
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
--prompt "a portrait of a professional person wearing formal attire, looking at the camera" \
--seed 1 \
--save_file ${OUTPUT_IMAGE}