envs='maze2d-umaze-v1 maze2d-medium-v1 maze2d-large-v1'

for env in $envs
do
    echo $env
    python data/download_d4rl_datasets.py --env_name=$env&
    wait
done
