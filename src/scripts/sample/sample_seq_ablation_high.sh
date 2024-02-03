

dataset="halfcheetah-medium-v2"
# ckpt8="results/halfcheetah-medium-v2/smallmixer_denoiser_v4_seq8/2024-01-02/04:49/model-1000000.pt"
# config8="smallmixer_denoiser_v4_seq8.yaml"

# dataset="hopper-medium-v2"
# ckpt="results/hopper-medium-v2/smallmixer_denoiser_v4/2023-12-23/model-1000000.pt"

# ckpt="/model/halfcheetah-model-1000000.pt"
config_name="smallmixer_denoiser_v4.yaml"

# dataset="walker2d-medium-v2"
# ckpt="results/walker2d-medium-v2/smallmixer_denoiser_v4/2023-12-23/model-1000000.pt"


# dataset="halfcheetah-medium-v2"
# ckpt="results/halfcheetah-medium-v2/smallmixer_denoiser_v4/2023-12-23/model-1000000.pt"

# dataset="hopper-medium-replay-v2"
# ckpt="results/hopper-medium-replay-v2/smallmixer_denoiser_v4/2023-12-30/18:45/model-1000000.pt"

dataset="walker2d-medium-replay-v2"
ckpt="results/walker2d-medium-replay-v2/smallmixer_denoiser_v4/2023-12-30/19:04/model-1000000.pt"

ext="npz"


temperature=2.0
CUDA_VISIBLE_DEVICES=0 \
accelerate launch src/diffusion/train_diffuser.py \
--load_checkpoint \
--dataset=$dataset \
--config_name=$config_name \
--ckpt_path=$ckpt \
--back_and_forth \
--noise_level=1.0 \
--temperature=$temperature \
--guidance_rewardscale=1.3 &\

CUDA_VISIBLE_DEVICES=2 \
accelerate launch src/diffusion/train_diffuser.py \
--load_checkpoint \
--dataset=$dataset \
--config_name=$config_name \
--ckpt_path=$ckpt \
--back_and_forth \
--noise_level=0.5 \
--temperature=$temperature \
--guidance_rewardscale=1.3 &\


wait
# for noise_level in 1.0
# do
#     CUDA_VISIBLE_DEVICES=0 \
#     accelerate launch src/diffusion/train_diffuser.py \
#     --load_checkpoint \
#     --dataset=$dataset\
#     --config_name=$config_name \
#     --ckpt_path=$ckpt \
#     --back_and_forth \
#     --noise_level=$noise_level \
#     --temperature=$temperature \
#     --guidance_rewardscale=1.3 &\

#     CUDA_VISIBLE_DEVICES=1 \
#     accelerate launch src/diffusion/train_diffuser.py \
#     --load_checkpoint \
#     --dataset=$dataset\
#     --config_name=$config_name \
#     --ckpt_path=$ckpt \
#     --back_and_forth \
#     --noise_level=$noise_level \
#     --temperature=$temperature \
#     --guidance_rewardscale=1.4 &\

#     wait
    
    
# done

wait
