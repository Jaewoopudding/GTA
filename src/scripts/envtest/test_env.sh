dataset='halfcheetah-medium-v2'
batch_num=12
ext="npz"


for file_nm in /home/sujin/project/Augmentation-For-OfflineRL/results/halfcheetah-medium-v2/smallmixer_denoiser_v4_reweighting_tune/2024-01-28/23:06/*
do
    echo $file_nm
    if [[ "$file_nm" == *"$ext"* ]]; then
        data_path=$file_nm
        echo $file_nm
        python src/scripts/envtest/aug_env_test.py \
        --dataset $dataset \
        --data_path $data_path \
        --batch_num $batch_num \
        --batch_id=0 &\
        python src/scripts/envtest/aug_env_test.py \
        --dataset $dataset \
        --data_path $data_path \
        --batch_num $batch_num \
        --batch_id=1 &\
        python src/scripts/envtest/aug_env_test.py \
        --dataset $dataset \
        --data_path $data_path \
        --batch_num $batch_num \
        --batch_id=2 &\
        python src/scripts/envtest/aug_env_test.py \
        --dataset $dataset \
        --data_path $data_path \
        --batch_num $batch_num \
        --batch_id=3 &\
        python src/scripts/envtest/aug_env_test.py \
        --dataset $dataset \
        --data_path $data_path \
        --batch_num $batch_num \
        --batch_id=4 &\
        python src/scripts/envtest/aug_env_test.py \
        --dataset $dataset \
        --data_path $data_path \
        --batch_num $batch_num \
        --batch_id=5 &\
        python src/scripts/envtest/aug_env_test.py \
        --dataset $dataset \
        --data_path $data_path \
        --batch_num $batch_num \
        --batch_id=6 &\
        python src/scripts/envtest/aug_env_test.py \
        --dataset $dataset \
        --data_path $data_path \
        --batch_num $batch_num \
        --batch_id=7 &\
        python src/scripts/envtest/aug_env_test.py \
        --dataset $dataset \
        --data_path $data_path \
        --batch_num $batch_num \
        --batch_id=8 &\
        python src/scripts/envtest/aug_env_test.py \
        --dataset $dataset \
        --data_path $data_path \
        --batch_num $batch_num \
        --batch_id=9 &\
        python src/scripts/envtest/aug_env_test.py \
        --dataset $dataset \
        --data_path $data_path \
        --batch_num $batch_num \
        --batch_id=10 &\
        python src/scripts/envtest/aug_env_test.py \
        --dataset $dataset \
        --data_path $data_path \
        --batch_num $batch_num \
        --batch_id=11 &
        wait
        python src/scripts/envtest/save_env_stats.py \
        --dataset $dataset \
        --data_path $data_path 
    else
        continue
    fi
done

wait 
