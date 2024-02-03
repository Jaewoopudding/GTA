config_name="smallmixer_denoiser_v4_maze.yaml"

dataset5="maze2d-large-v1"
ckpt5="/model/large-model-1000000.pt"

bash src/scripts/sample/sample_any_env_low.sh $dataset5 0 $ckpt5 $config_name&
bash src/scripts/sample/sample_any_env_high.sh $dataset5 1 $ckpt5 $config_name&


wait