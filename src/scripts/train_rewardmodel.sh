
for env_name in 'walker2d' 'halfcheetah' 'hopper'
do
    for dataset_type in 'medium-replay' 'medium' 'medium-expert'
    do
        echo "Training reward model for ${env_name}-${dataset_type}-v2"
        python /home/sujin/project/Augmentation-For-OfflineRL/src/dynamics/reward.py\
        --env "${env_name}-${dataset_type}-v2"
    done
done