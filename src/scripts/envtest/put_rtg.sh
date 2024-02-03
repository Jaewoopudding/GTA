# envs='hopper-medium-replay-v2 hopper-medium-expert-v2 walker2d-medium-expert-v2'
# envs='walker2d-medium-v2 halfcheetah-medium-expert-v2'
# envs='maze2d-large-v1 maze2d-medium-v1 maze2d-umaze-v1'
# envs='walker2d-medium-expert-v2'

# envs='halfcheetah-medium-replay-v2'
# halfcheetah-medium-v2 hopper-medium-v2 walker2d-medium-replay-v2'


envs='hopper-medium-expert-v2 walker2d-medium-expert-v2'

for env in $envs
do
    echo $env
    python /home/sujin/project/Augmentation-For-OfflineRL/src/scripts/envtest/put_RTG.py --env $env
done
