#!/bin/bash
#export PROJ_ROOT=$HOME/SparseAttention/sparse-attention
export CONFIG_ROOT=$PROJ_ROOT/configs

model_name=${1:-"bert-base-uncased"}
#model_config=${2:-"bert_base_sanger_2e-3.json"}
model_config_name=${2:-"adamax_nb80"}
model_config="gpt2_$model_config_name.json"
log_load_balance=${3:-false}
output_dir=${4:-"$PROJ_ROOT/outputs/squad"}

if [ "$log_load_balance" = true ] ; then
  export LOG_LOAD_BALANCE=true
fi
#export LOG_LOAD_BALANCE="true" # this controls logging 
export LOG_OPS_COUNT="true"
export TASK_NAME="SQUAD"
export model_config_name

python run_squad.py \
    --model_name_or_path sparse-$model_name \
    --config_name $CONFIG_ROOT/$model_config \
    --dataset_name squad \
    --do_eval \
    --max_seq_length 384 \
    --doc_stride 128 \
    --eval_checkpoint $output_dir/sparse-$model_config_name-$(basename $model_name)/pytorch_model.bin \
    --output_dir $output_dir/sparse-$model_config_name-$(basename $model_name)/
