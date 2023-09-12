#!/bin/bash

# cwd: /path/to/repo

#COMMAND_PREFIX="proxychains4 "
COMMAND_PREFIX=""

###################
# dense
###################
#cloth
#$COMMAND_PREFIX bash ./scripts/train_dense_on_cloth.sh 

#squad
#$COMMAND_PREFIX bash ./scripts/train_dense_on_squad.sh 

#glue
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh mrpc 
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh cola 
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh sst2 
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh qqp 
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh stsb 
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh mnli 
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh qnli 
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh rte 
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh wnli  

###################
# longformer
###################
#cloth
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased longformer

#squad
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_squad.sh bert-base-uncased longformer

#glue
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mrpc bert-base-uncased longformer
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh cola bert-base-uncased longformer
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh sst2 bert-base-uncased longformer
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qqp bert-base-uncased longformer
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh stsb bert-base-uncased longformer
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mnli bert-base-uncased longformer
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qnli bert-base-uncased longformer
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh rte bert-base-uncased longformer
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh wnli bert-base-uncased longformer

###################
# bigbird
###################

#cloth
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased bigbird

#squad
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_squad.sh bert-base-uncased bigbird

#glue
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mrpc bert-base-uncased bigbird
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh cola bert-base-uncased bigbird
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh sst2 bert-base-uncased bigbird
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qqp bert-base-uncased bigbird
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh stsb bert-base-uncased bigbird
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mnli bert-base-uncased bigbird
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qnli bert-base-uncased bigbird
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh rte bert-base-uncased bigbird
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh wnli bert-base-uncased bigbird

###################
# fixed
###################

#cloth
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased fixed

#squad
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_squad.sh bert-base-uncased fixed

#glue
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mrpc bert-base-uncased fixed
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh cola bert-base-uncased fixed
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh sst2 bert-base-uncased fixed
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qqp bert-base-uncased fixed
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh stsb bert-base-uncased fixed
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mnli bert-base-uncased fixed
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qnli bert-base-uncased fixed
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh rte bert-base-uncased fixed
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh wnli bert-base-uncased fixed

###################
# sanger_2e-2
###################

#cloth
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased sanger_2e-2

#squad
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_squad.sh bert-base-uncased sanger_2e-2

#glue
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mrpc bert-base-uncased sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh cola bert-base-uncased sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh sst2 bert-base-uncased sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qqp bert-base-uncased sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh stsb bert-base-uncased sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mnli bert-base-uncased sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qnli bert-base-uncased sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh rte bert-base-uncased sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh wnli bert-base-uncased sanger_2e-2

###################
# sanger_2e-3
###################

#cloth
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_cloth.sh bert-base-uncased sanger_2e-3

#squad
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_squad.sh bert-base-uncased sanger_2e-3

#glue
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mrpc bert-base-uncased sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh cola bert-base-uncased sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh sst2 bert-base-uncased sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qqp bert-base-uncased sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh stsb bert-base-uncased sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh mnli bert-base-uncased sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh qnli bert-base-uncased sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh rte bert-base-uncased sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/train_sparse_on_glue.sh wnli bert-base-uncased sanger_2e-3

###################
# dense - gpt2
###################
##cloth
#$COMMAND_PREFIX bash ./scripts/train_dense_on_cloth.sh gpt2
#
##squad
#$COMMAND_PREFIX bash ./scripts/train_dense_on_squad.sh gpt2
#
##glue
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh mrpc gpt2
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh cola gpt2
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh sst2 gpt2
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh qqp gpt2
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh stsb gpt2
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh mnli gpt2
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh qnli gpt2
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh rte gpt2
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh wnli  gpt2


###################
# sanger2e-3 - gpt2
###################
##cloth
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_cloth.sh gpt2 sanger_2e-3
#
##squad
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_squad.sh gpt2 sanger_2e-3
#
##glue
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh mrpc gpt2 sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh cola gpt2 sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh sst2 gpt2 sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh qqp gpt2 sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh stsb gpt2 sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh mnli gpt2 sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh qnli gpt2 sanger_2e-3
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh rte gpt2 sanger_2e-3
##$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh wnli  gpt2 sanger_2e-3


###################
# sanger2e-2 - gpt2
###################
##cloth
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_cloth.sh gpt2 sanger_2e-2
#
##squad
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_squad.sh gpt2 sanger_2e-2
#
##glue
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh mrpc gpt2 sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh cola gpt2 sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh sst2 gpt2 sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh qqp gpt2 sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh stsb gpt2 sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh mnli gpt2 sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh qnli gpt2 sanger_2e-2
#$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh rte gpt2 sanger_2e-2
##$COMMAND_PREFIX bash ./scripts/gpt2/train_sparse_on_glue.sh wnli  gpt2 sanger_2e-2


####################
## dense - facebook/bart-large
####################
##cloth
#$COMMAND_PREFIX bash ./scripts/train_dense_on_cloth.sh facebook/bart-large
#
##squad
#$COMMAND_PREFIX bash ./scripts/train_dense_on_squad.sh facebook/bart-large
#
##glue
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh mrpc facebook/bart-large
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh cola facebook/bart-large
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh sst2 facebook/bart-large
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh qqp facebook/bart-large
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh stsb facebook/bart-large
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh mnli facebook/bart-large
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh qnli facebook/bart-large
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh rte facebook/bart-large
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh wnli  facebook/bart-large


####################
## dense - facebook/bart-base
####################
##cloth
#$COMMAND_PREFIX bash ./scripts/train_dense_on_cloth.sh facebook/bart-base
#
##squad
#$COMMAND_PREFIX bash ./scripts/train_dense_on_squad.sh facebook/bart-base
#
##glue
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh mrpc facebook/bart-base
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh cola facebook/bart-base
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh sst2 facebook/bart-base
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh qqp facebook/bart-base
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh stsb facebook/bart-base
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh mnli facebook/bart-base
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh qnli facebook/bart-base
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh rte facebook/bart-base
#$COMMAND_PREFIX bash ./scripts/train_dense_on_glue.sh wnli  facebook/bart-base


###################
# sanger1e-2 - facebook/bart-base
###################
##cloth
#$COMMAND_PREFIX bash ./scripts/bart/train_sparse_on_cloth.sh facebook/bart-base sanger_1e-2
#
##squad
#$COMMAND_PREFIX bash ./scripts/bart/train_sparse_on_squad.sh facebook/bart-base sanger_1e-2
#
##glue
#$COMMAND_PREFIX bash ./scripts/bart/train_sparse_on_glue.sh mrpc facebook/bart-base sanger_1e-2
#$COMMAND_PREFIX bash ./scripts/bart/train_sparse_on_glue.sh cola facebook/bart-base sanger_1e-2
#$COMMAND_PREFIX bash ./scripts/bart/train_sparse_on_glue.sh sst2 facebook/bart-base sanger_1e-2
#$COMMAND_PREFIX bash ./scripts/bart/train_sparse_on_glue.sh qqp facebook/bart-base sanger_1e-2
#$COMMAND_PREFIX bash ./scripts/bart/train_sparse_on_glue.sh stsb facebook/bart-base sanger_1e-2
#$COMMAND_PREFIX bash ./scripts/bart/train_sparse_on_glue.sh mnli facebook/bart-base sanger_1e-2
#$COMMAND_PREFIX bash ./scripts/bart/train_sparse_on_glue.sh qnli facebook/bart-base sanger_1e-2
#$COMMAND_PREFIX bash ./scripts/bart/train_sparse_on_glue.sh rte facebook/bart-base sanger_1e-2
##$COMMAND_PREFIX bash ./scripts/bart/train_sparse_on_glue.sh wnli  facebook/bart-base sanger_1e-2


