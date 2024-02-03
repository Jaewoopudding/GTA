

dataset='halfcheetah-medium-expert-v2'
batch_num=10
ext="npz"
# 0.1 1.0 0.25 0.5 0.01
for percentage in 0
do
    for file_nm in /home/orl/temp0130/halfcheetah-medexp-ablation/med1/*
    do
        echo $file_nm
        if [[ "$file_nm" == *"$ext"* ]]; then
            data_path=$file_nm
            name=`basename $data_path`
            dataset=${name%.*}
            # dataset=${dataset%-*}
            echo $dataset
            echo $data_path
            python src/scripts/envtest/aug_env_test.py \
            --dataset=$dataset \
            --data_path=$data_path \
            --batch_num=$batch_num \
            --test_partial \
            --percentage=$percentage \
            --batch_id=0 &\
            python src/scripts/envtest/aug_env_test.py \
            --dataset=$dataset \
            --data_path=$data_path \
            --batch_num=$batch_num \
            --test_partial \
            --percentage=$percentage \
            --batch_id=1 &\
            python src/scripts/envtest/aug_env_test.py \
            --dataset=$dataset \
            --data_path=$data_path \
            --batch_num=$batch_num \
            --test_partial \
            --percentage=$percentage \
            --batch_id=2 &\
            python src/scripts/envtest/aug_env_test.py \
            --dataset=$dataset \
            --data_path=$data_path \
            --batch_num=$batch_num \
            --test_partial \
            --percentage=$percentage \
            --batch_id=3 &\
            python src/scripts/envtest/aug_env_test.py \
            --dataset=$dataset \
            --data_path=$data_path \
            --batch_num=$batch_num \
            --test_partial \
            --percentage=$percentage \
            --batch_id=4 &\
            python src/scripts/envtest/aug_env_test.py \
            --dataset=$dataset \
            --data_path=$data_path \
            --batch_num=$batch_num \
            --test_partial \
            --percentage=$percentage \
            --batch_id=5 &\
            python src/scripts/envtest/aug_env_test.py \
            --dataset=$dataset \
            --data_path=$data_path \
            --batch_num=$batch_num \
            --test_partial \
            --percentage=$percentage \
            --batch_id=6 &\
            python src/scripts/envtest/aug_env_test.py \
            --dataset=$dataset \
            --data_path=$data_path \
            --batch_num=$batch_num \
            --test_partial \
            --percentage=$percentage \
            --batch_id=7 &\
            python src/scripts/envtest/aug_env_test.py \
            --dataset=$dataset \
            --data_path=$data_path \
            --batch_num=$batch_num \
            --test_partial \
            --percentage=$percentage \
            --batch_id=8 &\
            python src/scripts/envtest/aug_env_test.py \
            --dataset=$dataset \
            --data_path=$data_path \
            --batch_num=$batch_num \
            --test_partial \
            --percentage=$percentage \
            --batch_id=9 &
            wait
            python src/scripts/envtest/save_env_stats.py \
            --dataset $dataset \
            --data_path $data_path \
            --percentage=$percentage 
        else
            continue
        fi
    done
done
wait 
