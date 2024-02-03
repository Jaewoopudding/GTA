
dataset='halfcheetah-medium-v2'
batch_num=12
ext="npz"


basepath=/home/orl/ablations-halfcheetah-medium-v2/envtest

for percentage in 1.0 0.5 0.1
do
    for file_nm in /home/orl/ablations-halfcheetah-medium-v2/envtest/*
    do
        echo $file_nm
        if [[ "$file_nm" == *"$ext"* ]]; then
            data_path=$file_nm
            name=`basename $data_path`
            # dataset=${name%.*}
            # dataset=${dataset%-*}
            echo $dataset
            echo $data_path
            python /home/sujin/project/Augmentation-For-OfflineRL/src/scripts/envtest/mmd_test_traj.py \
            --dataset=$dataset \
            --data_path=$basepath \
            --test_data_nm=$name \
            --test_partial \
            --percentage=$percentage \
            --kernel=rbf \
            --device=cuda:6 \

            wait
        fi
    done
done


