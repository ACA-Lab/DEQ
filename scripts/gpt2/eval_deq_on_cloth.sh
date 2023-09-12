#!/bin/bash
export CONFIG_ROOT=$PROJ_ROOT/configs

model_name=${1:-"bert-base-uncased"}
#model_config=${2:-"bert_base_sanger_2e-3.json"}
model_config_name=${2:-"adamax_nb80"}
model_config="gpt2_$model_config_name.json"
num_train_epochs=${3:-"20"}
log_load_balance=${4:-false}
output_dir=${5:-"$PROJ_ROOT/outputs/cloth"}

if [ "$log_load_balance" = true ] ; then
  export LOG_LOAD_BALANCE=true
fi
#export LOG_LOAD_BALANCE="true"  # this controls logging 
export LOG_OPS_COUNT="true"
export TASK_NAME="CLOTH"
export model_config_name

python run_cloth.py \
    --model_name_or_path sparse-$model_name \
    --config_name $CONFIG_ROOT/$model_config \
    --do_eval \
    --max_seq_length 512 \
    --num_train_epochs $num_train_epochs \
    --eval_checkpoint $output_dir/sparse-$model_config_name-$(basename $model_name)/checkpoint_${num_train_epochs} \
    --output_dir $output_dir/sparse-$model_config_name-$(basename $model_name)/
