CUDA_VISIBLE_DEVICES="4" \
accelerate launch src/diffusion/train_diffuser.py \
--load_checkpoint \
--ckpt_path=/home/sujin/project/Augmentation-For-OfflineRL/results/halfcheetah-medium-v2/mixer_denoiser/2023-11-17/18:15-cond-sar/model-1000000.pt \
--back_and_forth \
--config_name=mixer_denoiser.yaml

# note : reward scale need to be 1.0
# it scales to 0.001 after update on 1127
# construct_diffusion_model.denoising_network.dim = 32