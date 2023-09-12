#!/bin/bash
export CONFIG_ROOT=$PROJ_ROOT/configs

model_name=${1:-"bert-base-uncased"}
#model_config=${2:-"bert_base_sanger_2e-3.json"}
model_config_name=${2:-"sanger_2e-3"}
model_config="bert_base_$model_config_name.json"
num_train_epochs=${3:-"20"}
learning_rate=${4:-"5e-5"}
batch_size=${5:-"3"}
output_dir=${6:-"$PROJ_ROOT/outputs/cloth"}

export LOG_LOAD_BALANCE="true"
export LOG_OPS_COUNT="true"
export TASK_NAME="CLOTH"
export model_config_name

python run_cloth.py \
    --model_name_or_path sparse-$model_name \
    --config_name $CONFIG_ROOT/$model_config \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --train_batch_size $batch_size \
    --learning_rate $learning_rate \
    --num_train_epochs $num_train_epochs \
    --output_dir $output_dir/sparse-$model_config_name-$(basename $model_name)/