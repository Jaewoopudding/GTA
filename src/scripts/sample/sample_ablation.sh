

dataset="halfcheetah-medium-v2"
# ckpt8="results/halfcheetah-medium-v2/smallmixer_denoiser_v4_seq8/2024-01-02/04:49/model-1000000.pt"
# config8="smallmixer_denoiser_v4_seq8.yaml"

ckpt16="/model/seq64-model-1000000.pt"
config16="smallmixer_denoiser_v4_seq64.yaml"


ext="npz"

ckpt=$ckpt16
config_name=$config16


for noise_level in 0.5 
do

    CUDA_VISIBLE_DEVICES=0 \
    accelerate launch src/diffusion/train_diffuser.py \
    --load_checkpoint \
    --dataset=$dataset\
    --config_name=$config_name \
    --ckpt_path=$ckpt \
    --back_and_forth \
    --noise_level=$noise_level \
    --temperature=2.0 \
    --guidance_rewardscale=1.25 &\

    CUDA_VISIBLE_DEVICES=1 \
    accelerate launch src/diffusion/train_diffuser.py \
    --load_checkpoint \
    --dataset=$dataset\
    --config_name=$config_name \
    --ckpt_path=$ckpt \
    --back_and_forth \
    --noise_level=$noise_level \
    --temperature=2. \
    --guidance_rewardscale=1.3 &

    wait
    
    
done

wait
