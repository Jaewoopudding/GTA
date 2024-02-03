
config_name1="smallmixer_denoiser_v4_reweighting.yaml"
config_name2="smallmixer_denoiser_v4.yaml"

dataset1="hopper-random-v2"
ckpt1="results/hopper-random-v2/smallmixer_denoiser_v4/2024-01-15/03:20/model-1000000.pt"

# dataset2="halfcheetah-random-v2"
# ckpt2="results/halfcheetah-random-v2/smallmixer_denoiser_v4/2024-01-15/03:18/model-1000000.pt"

dataset3="walker2d-random-v2"
ckpt3="results/walker2d-random-v2/smallmixer_denoiser_v4/2024-01-15/07:00/model-1000000.pt"



# bash src/scripts/sample/sample_any_env_final.sh $dataset1 0 $ckpt1 $config_name1&
# bash src/scripts/sample/sample_any_env_final.sh $dataset1 1 $ckpt1 $config_name2&
# bash src/scripts/sample/sample_any_env_final.sh $dataset3 2 $ckpt3 $config_name1&
bash src/scripts/sample/sample_any_env_final.sh $dataset3 2 $ckpt3 $config_name2&