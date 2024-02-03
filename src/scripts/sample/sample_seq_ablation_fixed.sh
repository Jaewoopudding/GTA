

dataset="halfcheetah-medium-v2"
# ckpt8="results/halfcheetah-medium-v2/smallmixer_denoiser_v4_seq8/2024-01-02/04:49/model-1000000.pt"
# config8="smallmixer_denoiser_v4_seq8.yaml"

ckpt="/model/halfcheetah-model-1000000.pt"
config_name="smallmixer_denoiser_v4.yaml"


ext="npz"


temperature=2.0

for noise_level in 0.1 0.25 0.5 0.75 1.0
do
    CUDA_VISIBLE_DEVICES=0 \
    accelerate launch src/diffusion/train_diffuser.py \
    --load_checkpoint \
    --dataset=$dataset\
    --config_name=$config_name \
    --ckpt_path=$ckpt \
    --back_and_forth \
    --noise_level=$noise_level \
    --temperature=$temperature \
    --fixed_rewardscale=1.3 &\

    CUDA_VISIBLE_DEVICES=1 \
    accelerate launch src/diffusion/train_diffuser.py \
    --load_checkpoint \
    --dataset=$dataset\
    --config_name=$config_name \
    --ckpt_path=$ckpt \
    --back_and_forth \
    --noise_level=$noise_level \
    --temperature=$temperature \
    --fixed_rewardscale=1.0 &\

    wait
    
done

wait
