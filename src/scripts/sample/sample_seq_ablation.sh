

dataset="halfcheetah-medium-v2"
# ckpt8="results/halfcheetah-medium-v2/smallmixer_denoiser_v4_seq8/2024-01-02/04:49/model-1000000.pt"
# config8="smallmixer_denoiser_v4_seq8.yaml"

ckpt16="/model/seq1-model-1000000.pt"
config1="smallmixer_denoiser_v4_seq1.yaml"
config2="smallmixer_denoiser_v4_seq1_reweighting.yaml"


ext="npz"

ckpt=$ckpt16


for noise_level in 0.5 
do

    CUDA_VISIBLE_DEVICES=0 \
    accelerate launch src/diffusion/train_diffuser.py \
    --load_checkpoint \
    --dataset=$dataset\
    --config_name=$config1 \
    --ckpt_path=$ckpt \
    --back_and_forth \
    --noise_level=$noise_level \
    --temperature=2.0 \
    --guidance_rewardscale=1.3 &\

    CUDA_VISIBLE_DEVICES=1 \
    accelerate launch src/diffusion/train_diffuser.py \
    --load_checkpoint \
    --dataset=$dataset\
    --config_name=$config2 \
    --ckpt_path=$ckpt \
    --back_and_forth \
    --noise_level=$noise_level \
    --temperature=2.0 \
    --guidance_rewardscale=1.3 &

    wait
    
    
done

wait
