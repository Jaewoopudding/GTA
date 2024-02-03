config_name="smallmixer_denoiser_v4_maze.yaml"

dataset1="maze2d-large-v1"
ckpt1="results/maze2d-large-v1/smallmixer_denoiser_v4_maze/2024-01-04/06:45/model-1000000.pt"

dataset2="maze2d-medium-v1"
ckpt2="results/maze2d-medium-v1/smallmixer_denoiser_v4_maze/2024-01-04/01:14/model-1000000.pt"

dataset3="maze2d-umaze-v1"
ckpt3="results/maze2d-umaze-v1/smallmixer_denoiser_v4_maze/2024-01-04/01:14/model-1000000.pt"

bash src/scripts/sample/sample_any_env_final.sh $dataset1 4 $ckpt1 $config_name&\
bash src/scripts/sample/sample_any_env_final.sh $dataset2 5 $ckpt2 $config_name&\
bash src/scripts/sample/sample_any_env_final.sh $dataset3 6 $ckpt3 $config_name&\

wait