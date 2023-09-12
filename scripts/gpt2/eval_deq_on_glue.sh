#!/bin/bash
export CONFIG_ROOT=$PROJ_ROOT/configs

task_name=${1:-"mrpc"}
model_name=${2:-"bert-base-uncased"}
#model_config=${3:-"bert_base_sanger_2e-2.json"}
model_config_name=${3:-"adamax_nb80"}
model_config="gpt2_$model_config_name.json"
log_load_balance=${4:-false}
output_dir=${5:-"$PROJ_ROOT/outputs/glue"}

if [ "$log_load_balance" = true ] ; then
  export LOG_LOAD_BALANCE=true
fi
#export LOG_LOAD_BALANCE="true"  # this controls logging 
export LOG_OPS_COUNT="true"
export TASK_NAME="GLUE_$task_name"
export model_config_name

python run_glue.py \
    --model_name_or_path sparse-$model_name \
    --config_name $CONFIG_ROOT/$model_config \
    --task_name $task_name \
    --do_eval \
    --max_seq_length 128 \
    --eval_checkpoint $output_dir/sparse-$model_config_name-$(basename $model_name)_$task_name/pytorch_model.bin \
    --output_dir $output_dir/sparse-$model_config_name-$(basename $model_name)_$task_name/
