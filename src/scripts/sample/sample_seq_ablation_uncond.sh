

dataset="halfcheetah-medium-v2"
# ckpt8="results/halfcheetah-medium-v2/smallmixer_denoiser_v4_seq8/2024-01-02/04:49/model-1000000.pt"
# config8="smallmixer_denoiser_v4_seq8.yaml"

ckpt="/model/halfcheetah-model-1000000.pt"
config_name="smallmixer_denoiser_v4.yaml"


ext="npz"


temperature=0.0

CUDA_VISIBLE_DEVICES=0 \
accelerate launch src/diffusion/train_diffuser.py \
--load_checkpoint \
--dataset=$dataset\
--config_name=$config_name \
--ckpt_path=$ckpt \
--back_and_forth \
--noise_level=0.1 \
--temperature=$temperature \
--guidance_rewardscale=1.0 &\

CUDA_VISIBLE_DEVICES=1 \
accelerate launch src/diffusion/train_diffuser.py \
--load_checkpoint \
--dataset=$dataset\
--config_name=$config_name \
--ckpt_path=$ckpt \
--back_and_forth \
--noise_level=0.25 \
--temperature=$temperature \
--guidance_rewardscale=1.0 &\

wait

CUDA_VISIBLE_DEVICES=0 \
accelerate launch src/diffusion/train_diffuser.py \
--load_checkpoint \
--dataset=$dataset\
--config_name=$config_name \
--ckpt_path=$ckpt \
--back_and_forth \
--noise_level=0.5 \
--temperature=$temperature \
--guidance_rewardscale=1.0 &\

CUDA_VISIBLE_DEVICES=1 \
accelerate launch src/diffusion/train_diffuser.py \
--load_checkpoint \
--dataset=$dataset\
--config_name=$config_name \
--ckpt_path=$ckpt \
--back_and_forth \
--noise_level=0.75 \
--temperature=$temperature \
--guidance_rewardscale=1.0 &\

wait
CUDA_VISIBLE_DEVICES=0 \
accelerate launch src/diffusion/train_diffuser.py \
--load_checkpoint \
--dataset=$dataset\
--config_name=$config_name \
--ckpt_path=$ckpt \
--back_and_forth \
--noise_level=1.0 \
--temperature=$temperature \
--guidance_rewardscale=1.0 &\

wait