#!/bin/bash

# cwd: /path/to/repo

#COMMAND_PREFIX="proxychains4 "
COMMAND_PREFIX=""
export WANDB_DISABLED=true

###################
# refine 2e-2
###################

##cloth
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased refine_2e-2
#
##squad
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_squad.sh bert-base-uncased refine_2e-2
#
##glue
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mrpc bert-base-uncased refine_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh cola bert-base-uncased refine_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh sst2 bert-base-uncased refine_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qqp bert-base-uncased refine_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh stsb bert-base-uncased refine_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mnli bert-base-uncased refine_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qnli bert-base-uncased refine_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh rte bert-base-uncased refine_2e-2
##$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh wnli bert-base-uncased refine_2e-2


###################
# refine 2e-3
###################

##cloth
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased refine_2e-3
#
##squad
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_squad.sh bert-base-uncased refine_2e-3
#
##glue
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mrpc bert-base-uncased refine_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh cola bert-base-uncased refine_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh sst2 bert-base-uncased refine_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qqp bert-base-uncased refine_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh stsb bert-base-uncased refine_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mnli bert-base-uncased refine_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qnli bert-base-uncased refine_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh rte bert-base-uncased refine_2e-3
##$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh wnli bert-base-uncased refine_2e-3


############################################################
#cloth
#$COMMAND_PREFIX bash ./scripts/train_dense_on_cloth.sh 
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased longformer
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased bigbird
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased fixed
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased sanger_2e-3


##################################################

## eval

# cloth
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_cloth.sh bert-base-uncased sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_cloth.sh bert-base-uncased sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_cloth.sh bert-base-uncased refine_2e-2
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_cloth.sh bert-base-uncased refine_2e-3

## squad
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_squad.sh bert-base-uncased sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_squad.sh bert-base-uncased refine_2e-2
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_squad.sh bert-base-uncased sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_squad.sh bert-base-uncased refine_2e-3

## glue
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh mrpc bert-base-uncased refine_2e-2
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh cola bert-base-uncased refine_2e-2
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh sst2 bert-base-uncased refine_2e-2
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh qqp bert-base-uncased refine_2e-2
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh stsb bert-base-uncased refine_2e-2
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh mnli bert-base-uncased refine_2e-2
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh qnli bert-base-uncased refine_2e-2
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh rte bert-base-uncased refine_2e-2
##$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh wnli bert-base-uncased refine_2e-2
#
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh mrpc bert-base-uncased sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh cola bert-base-uncased sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh sst2 bert-base-uncased sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh qqp bert-base-uncased sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh stsb bert-base-uncased sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh mnli bert-base-uncased sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh qnli bert-base-uncased sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh rte bert-base-uncased sanger_2e-2
##$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh wnli bert-base-uncased sanger_2e-2
#
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh mrpc bert-base-uncased sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh cola bert-base-uncased sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh sst2 bert-base-uncased sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh qqp bert-base-uncased sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh stsb bert-base-uncased sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh mnli bert-base-uncased sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh qnli bert-base-uncased sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh rte bert-base-uncased sanger_2e-3
##$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh wnli bert-base-uncased sanger_2e-3
#
#
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh mrpc bert-base-uncased refine_2e-3
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh cola bert-base-uncased refine_2e-3
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh sst2 bert-base-uncased refine_2e-3
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh qqp bert-base-uncased refine_2e-3
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh stsb bert-base-uncased refine_2e-3
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh mnli bert-base-uncased refine_2e-3
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh qnli bert-base-uncased refine_2e-3
#$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh rte bert-base-uncased refine_2e-3
##$COMMAND_PREFIX bash ./scripts/my_eval_sparse_on_glue.sh wnli bert-base-uncased refine_2e-3
############################

###################
#refine inv 2e-3
###################

##cloth
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased refine_inv_2e-3
#
##squad
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_squad.sh bert-base-uncased refine_inv_2e-3
#
##glue
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mrpc bert-base-uncased refine_inv_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh cola bert-base-uncased refine_inv_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh sst2 bert-base-uncased refine_inv_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qqp bert-base-uncased refine_inv_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh stsb bert-base-uncased refine_inv_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mnli bert-base-uncased refine_inv_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qnli bert-base-uncased refine_inv_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh rte bert-base-uncased refine_inv_2e-3
##$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh wnli bert-base-uncased refine_inv_2e-3

###################
#refine inv 2e-2
###################

#cloth
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased refine_inv_2e-2

##squad
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_squad.sh bert-base-uncased refine_inv_2e-2
#
##glue
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mrpc bert-base-uncased refine_inv_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh cola bert-base-uncased refine_inv_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh sst2 bert-base-uncased refine_inv_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qqp bert-base-uncased refine_inv_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh stsb bert-base-uncased refine_inv_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mnli bert-base-uncased refine_inv_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qnli bert-base-uncased refine_inv_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh rte bert-base-uncased refine_inv_2e-2
##$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh wnli bert-base-uncased refine_inv_2e-2

###################
# refine 8e-2
###################

##cloth
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased refine_8e-2
#
##squad
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_squad.sh bert-base-uncased refine_8e-2
#
##glue
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mrpc bert-base-uncased refine_8e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh cola bert-base-uncased refine_8e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh sst2 bert-base-uncased refine_8e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qqp bert-base-uncased refine_8e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh stsb bert-base-uncased refine_8e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mnli bert-base-uncased refine_8e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qnli bert-base-uncased refine_8e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh rte bert-base-uncased refine_8e-2
##$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh wnli bert-base-uncased refine_8e-2

###################
# refine 2e-1
###################

##cloth
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased refine_2e-1
#
##squad
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_squad.sh bert-base-uncased refine_2e-1
#
##glue
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mrpc bert-base-uncased refine_2e-1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh cola bert-base-uncased refine_2e-1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh sst2 bert-base-uncased refine_2e-1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qqp bert-base-uncased refine_2e-1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh stsb bert-base-uncased refine_2e-1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mnli bert-base-uncased refine_2e-1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qnli bert-base-uncased refine_2e-1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh rte bert-base-uncased refine_2e-1
##$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh wnli bert-base-uncased refine_2e-1


###################
# refine 8e-1
###################

##cloth
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased refine_8e-1
#
##squad
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_squad.sh bert-base-uncased refine_8e-1
#
##glue
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mrpc bert-base-uncased refine_8e-1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh cola bert-base-uncased refine_8e-1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh sst2 bert-base-uncased refine_8e-1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qqp bert-base-uncased refine_8e-1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh stsb bert-base-uncased refine_8e-1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mnli bert-base-uncased refine_8e-1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qnli bert-base-uncased refine_8e-1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh rte bert-base-uncased refine_8e-1
##$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh wnli bert-base-uncased refine_8e-1


###################
# refine 2e1
###################

##cloth
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased refine_2e1
#
##squad
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_squad.sh bert-base-uncased refine_2e1
#
##glue
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mrpc bert-base-uncased refine_2e1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh cola bert-base-uncased refine_2e1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh sst2 bert-base-uncased refine_2e1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qqp bert-base-uncased refine_2e1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh stsb bert-base-uncased refine_2e1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mnli bert-base-uncased refine_2e1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qnli bert-base-uncased refine_2e1
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh rte bert-base-uncased refine_2e1
##$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh wnli bert-base-uncased refine_2e1


###################
# adamax h50
###################

##cloth
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased adamax_h50
#
##squad
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_squad.sh bert-base-uncased adamax_h50
#
##glue
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mrpc bert-base-uncased adamax_h50
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh cola bert-base-uncased adamax_h50
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh sst2 bert-base-uncased adamax_h50
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qqp bert-base-uncased adamax_h50
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh stsb bert-base-uncased adamax_h50
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mnli bert-base-uncased adamax_h50
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qnli bert-base-uncased adamax_h50
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh rte bert-base-uncased adamax_h50
##$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh wnli bert-base-uncased adamax_h50

###################
# adamax h80
###################

##cloth
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased adamax_h80
#
##squad
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_squad.sh bert-base-uncased adamax_h80
#
##glue
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mrpc bert-base-uncased adamax_h80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh cola bert-base-uncased adamax_h80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh sst2 bert-base-uncased adamax_h80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qqp bert-base-uncased adamax_h80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh stsb bert-base-uncased adamax_h80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mnli bert-base-uncased adamax_h80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qnli bert-base-uncased adamax_h80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh rte bert-base-uncased adamax_h80
##$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh wnli bert-base-uncased adamax_h80

###################
# adamax s80
###################

##cloth
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased adamax_s80
#
##squad
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_squad.sh bert-base-uncased adamax_s80
#
##glue
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mrpc bert-base-uncased adamax_s80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh cola bert-base-uncased adamax_s80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh sst2 bert-base-uncased adamax_s80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qqp bert-base-uncased adamax_s80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh stsb bert-base-uncased adamax_s80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mnli bert-base-uncased adamax_s80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qnli bert-base-uncased adamax_s80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh rte bert-base-uncased adamax_s80
##$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh wnli bert-base-uncased adamax_s80

###################
# adamax ns80
###################

##cloth
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased adamax_ns80
#
##squad
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_squad.sh bert-base-uncased adamax_ns80
#
##glue
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mrpc bert-base-uncased adamax_ns80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh cola bert-base-uncased adamax_ns80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh sst2 bert-base-uncased adamax_ns80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qqp bert-base-uncased adamax_ns80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh stsb bert-base-uncased adamax_ns80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mnli bert-base-uncased adamax_ns80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qnli bert-base-uncased adamax_ns80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh rte bert-base-uncased adamax_ns80
##$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh wnli bert-base-uncased adamax_ns80


###################
# refine_8e-1 - gpt2
###################
##cloth
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_cloth.sh gpt2 refine_8e-1
#
##squad
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_squad.sh gpt2 refine_8e-1
#
##glue
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh mrpc gpt2 refine_8e-1
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh cola gpt2 refine_8e-1
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh sst2 gpt2 refine_8e-1
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh qqp gpt2 refine_8e-1
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh stsb gpt2 refine_8e-1
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh mnli gpt2 refine_8e-1
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh qnli gpt2 refine_8e-1
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh rte gpt2 refine_8e-1
##$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh wnli  gpt2 refine_8e-1

###################
# refine_2e1 - gpt2
###################

##glue
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh mrpc gpt2 refine_2e1
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh sst2 gpt2 refine_2e1
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh qqp gpt2 refine_2e1
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh stsb gpt2 refine_2e1
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh mnli gpt2 refine_2e1
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh qnli gpt2 refine_2e1
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh rte gpt2 refine_2e1

###################
# refine_2e-1 - gpt2
###################

##glue
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh mrpc gpt2 refine_2e-1
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh sst2 gpt2 refine_2e-1
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh qqp gpt2 refine_2e-1
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh stsb gpt2 refine_2e-1
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh mnli gpt2 refine_2e-1
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh qnli gpt2 refine_2e-1
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh rte gpt2 refine_2e-1

###################
# refine_2e-2 - gpt2
###################

##glue
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh mrpc gpt2 refine_2e-2
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh sst2 gpt2 refine_2e-2
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh qqp gpt2 refine_2e-2
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh stsb gpt2 refine_2e-2
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh mnli gpt2 refine_2e-2
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh qnli gpt2 refine_2e-2
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh rte gpt2 refine_2e-2

###################
# refine_2e-3 - gpt2
###################

##glue
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh mrpc gpt2 refine_2e-3
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh sst2 gpt2 refine_2e-3
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh qqp gpt2 refine_2e-3
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh stsb gpt2 refine_2e-3
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh mnli gpt2 refine_2e-3
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh qnli gpt2 refine_2e-3
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh rte gpt2 refine_2e-3


###################
# adamax b80
###################

##cloth
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased adamax_b80
#
##squad
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_squad.sh bert-base-uncased adamax_b80
#
##glue
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mrpc bert-base-uncased adamax_b80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh cola bert-base-uncased adamax_b80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh sst2 bert-base-uncased adamax_b80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qqp bert-base-uncased adamax_b80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh stsb bert-base-uncased adamax_b80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mnli bert-base-uncased adamax_b80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qnli bert-base-uncased adamax_b80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh rte bert-base-uncased adamax_b80
##$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh wnli bert-base-uncased adamax_b80

###################
# adamax nb80
###################

##cloth
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased adamax_nb80
#
##squad
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_squad.sh bert-base-uncased adamax_nb80
#
##glue
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mrpc bert-base-uncased adamax_nb80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh cola bert-base-uncased adamax_nb80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh sst2 bert-base-uncased adamax_nb80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qqp bert-base-uncased adamax_nb80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh stsb bert-base-uncased adamax_nb80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mnli bert-base-uncased adamax_nb80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qnli bert-base-uncased adamax_nb80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh rte bert-base-uncased adamax_nb80
##$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh wnli bert-base-uncased adamax_nb80

###################
# adamax nh80
###################

##cloth
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased adamax_nh80
#
##squad
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_squad.sh bert-base-uncased adamax_nh80
#
##glue
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mrpc bert-base-uncased adamax_nh80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh cola bert-base-uncased adamax_nh80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh sst2 bert-base-uncased adamax_nh80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qqp bert-base-uncased adamax_nh80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh stsb bert-base-uncased adamax_nh80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mnli bert-base-uncased adamax_nh80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qnli bert-base-uncased adamax_nh80
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh rte bert-base-uncased adamax_nh80
##$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh wnli bert-base-uncased adamax_nh80


