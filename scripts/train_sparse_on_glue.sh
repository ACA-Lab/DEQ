#!/bin/bash
export CONFIG_ROOT=$PROJ_ROOT/configs

task_name=${1:-"mrpc"}
model_name=${2:-"bert-base-uncased"}
# model_config=${3:-"bert_base_sanger_2e-2.json"}
model_config_name=${3:-"sanger_2e-3"}
model_config="bert_base_$model_config_name.json"
num_train_epochs=${4:-"3"}
learning_rate=${5:-"2e-5"}
batch_size=${6:-"32"}
output_dir=${7:-"$PROJ_ROOT/outputs/glue"}

export WANDB_DISABLED="true"
export LOG_LOAD_BALANCE="true"
export LOG_OPS_COUNT="true"
export TASK_NAME="glue_$task_name"
export model_config_name

python run_glue.py \
    --model_name_or_path sparse-$model_name \
    --config_name $CONFIG_ROOT/$model_config \
    --task_name $task_name \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size $batch_size \
    --learning_rate $learning_rate \
    --num_train_epochs $num_train_epochs \
    --output_dir $output_dir/sparse-$model_config_name-$(basename $model_name)_$task_name/
