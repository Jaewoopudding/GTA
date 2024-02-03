CUDA_VISIBLE_DEVICES="3" \
accelerate launch src/diffusion/train_diffuser.py \
--load_checkpoint \
--ckpt_path=/home/sujin/project/Augmentation-For-OfflineRL/results/halfcheetah-medium-v2/temporalattention_denoiser/2023-11-17/18:19_sar/model-1000000.pt \
--back_and_forth \
--config_name=temporalattention_denoiser.yaml


# SimpleDiffusionGenerator.noise_level = 1.0