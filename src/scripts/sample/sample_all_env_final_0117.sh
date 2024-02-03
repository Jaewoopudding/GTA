
config_name1="smallmixer_denoiser_v4_reweighting.yaml"

# dataset1="hopper-medium-replay-v2"
# ckpt1="results/hopper-medium-replay-v2/smallmixer_denoiser_v4/2023-12-30/18:45/model-1000000.pt"

dataset2="hopper-medium-v2"
ckpt2="results/hopper-medium-v2/smallmixer_denoiser_v4/2023-12-23/model-1000000.pt"

# dataset3="hopper-medium-expert-v2"
# ckpt3="results/hopper-medium-expert-v2/smallmixer_denoiser_v4/2024-01-07/18:59/model-1000000.pt"

# dataset4="walker2d-medium-replay-v2"
# ckpt4="results/walker2d-medium-replay-v2/smallmixer_denoiser_v4/2023-12-30/19:04/model-1000000.pt"

dataset5="walker2d-medium-v2"
ckpt5="results/walker2d-medium-v2/smallmixer_denoiser_v4/2023-12-23/model-1000000.pt"

# dataset6="walker2d-medium-expert-v2"
# ckpt6="results/walker2d-medium-expert-v2/smallmixer_denoiser_v4/2024-01-11/12:49/model-1000000.pt"

# dataset7="halfcheetah-medium-replay-v2"
# ckpt7="results/halfcheetah-medium-replay-v2/smallmixer_denoiser_v4/2024-01-07/06:34/model-1000000.pt"

dataset8="halfcheetah-medium-v2"
ckpt8="results/halfcheetah-medium-v2/smallmixer_denoiser_v4/2023-12-23/model-1000000.pt"

# dataset9="halfcheetah-medium-expert-v2"
# ckpt9="results/halfcheetah-medium-expert-v2/smallmixer_denoiser_v4/2024-01-11/12:31/model-1000000.pt"



config_name2="smallmixer_denoiser_v4_maze.yaml"

dataset10="maze2d-large-v1"
ckpt10="results/maze2d-large-v1/smallmixer_denoiser_v4_maze/2024-01-04/06:45/model-1000000.pt"

dataset11="maze2d-medium-v1"
ckpt11="results/maze2d-medium-v1/smallmixer_denoiser_v4_maze/2024-01-04/01:14/model-1000000.pt"

# dataset12="maze2d-umaze-v1"
# ckpt12="results/maze2d-umaze-v1/smallmixer_denoiser_v4_maze/2024-01-04/01:14/model-1000000.pt"




# # bash src/scripts/sample/sample_any_env_final.sh $dataset1 0 $ckpt1 $config_name1&

# bash src/scripts/sample/sample_any_env_final.sh $dataset3 2 $ckpt3 $config_name1&
# # bash src/scripts/sample/sample_any_env_final.sh $dataset4 3 $ckpt4 $config_name1&


# wait
# maze 10x!!

# bash src/scripts/sample/sample_any_env_final.sh $dataset6 0 $ckpt6 $config_name1&
# bash src/scripts/sample/sample_any_env_final.sh $dataset7 1 $ckpt7 $config_name1&
# bash src/scripts/sample/sample_any_env_final.sh $dataset8 0 $ckpt8 $config_name1&
# bash src/scripts/sample/sample_any_env_final2.sh $dataset8 1 $ckpt8 $config_name1&
# # bash src/scripts/sample/sample_any_env_final.sh $dataset9 3 $ckpt9 $config_name1&
# bash src/scripts/sample/sample_any_env_maze.sh $dataset10 5 $ckpt10 $config_name2&
bash src/scripts/sample/sample_any_env_final2.sh $dataset2 4 $ckpt2 $config_name1&
# wait
# bash src/scripts/sample/sample_any_env_maze.sh $dataset11 7 $ckpt11 $config_name2&
bash src/scripts/sample/sample_any_env_final2.sh $dataset5 2 $ckpt5 $config_name1&
# wait
# bash src/scripts/sample/sample_any_env_final.sh $dataset12 1 $ckpt12 $config_name2&
# bash src/scripts/sample/sample_any_env_final.sh $dataset3 3 $ckpt3 $config_name1&
# wait


