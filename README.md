# DEQ

This repository implements the proposed framework in the paper DEQ: Dynamic Element-wise Quantization for Efficient Attention Architecture (ICCD'23)

## Getting Started

### Requirements

-  For software experiments
   -  CUDA SDK >= 10.1
   -  Python >= 3.7
   -  PyTorch >= 1.7.0
   -  :hugs: Transformers 4.7.0

### Installation

1.  Clone or download this repository
2.  Download the CLOTH dataset from [here](https://www.cs.cmu.edu/~glai1/data/cloth/) to `data/cloth`
3.  Create a virtual environment (either [virtualenv](https://virtualenv.pypa.io/en/latest/) or [conda](https://docs.anaconda.com/anaconda/install/index.html)) with a Python version of at least 3.7.
4.  Install dependent Python packages: `pip install -r requirements.txt`
5.  Set relevant environment variables
    1.  `export PROJ_ROOT=<path-to-this-repo>`
    2.  `export WANDB_ENABLED=true` to enable [wandb](https://docs.wandb.ai/quickstart) logging (optional)

## Experiment Workflow

### Software experiments

1.  Evaluate DEQ performance

    1.  Train a model. 

        We provide scripts for training in the `scripts/` sub-directory. For example, to train a DEQ-pruned BERT-Base model on GLUE for MRPC, you can execute `scripts/train_sparse_on_glue.sh mrpc bert-base adamax_nb80`. Note that you have to pass in an appropriate configuration file, which you can find in `configs/`. You can skip this step if you choose to load a fine-tuned checkpoint directly.

    2.  Evaluate the fine-tuned model. 

        We also provide scripts for evaluation in `scripts/`. For example, to evaluate the sparse model from the last step, you can execute `scripts/eval_deq_on_glue.sh mrpc bert-base adamax_nb80`. If you need to load a checkpoint from a non-standard location, be sure to change the path in the script. When the evaluation is complete, the script should print out the accuracy.

2.  Comparison with dense attention and static sparse attention.

    1.  Train a model with dense or static sparse attention. 

        We provide dedicated scripts for train models with dense attention (e.g. `scripts/train_dense_on_squad.sh`). To train a model with static sparse attention, you can use the same script as DEQ and pass in an appropriate configuration file (e.g. `bert_base_longformer.json`).

    2.  Evaluate the fine-tuned model. 

        The process is similar to evaluating DEQ models. Note that you also need to use different scripts when evaluating dense models.

3.  Comparison with CPU and GPU.

    You can measure the latency of dense attention on CPU and GPU by executing `bench_cpu_gpu.py`.



## Citation

Xuhang Wang, Qiyue Huang, Zhuoran Song and Xiaoyao Liang. DEQ: Dynamic Element-wise Quantization for Efficient Attention Architecture (ICCD'23)

## Reference

DEQ is built on top of Sanger by Liqiang Lu. We thank the authors for releasing their code. If you use our model, please consider citing Sanger as well:


```BibTeX
 @inproceedings{lu2021sanger,
 title={Sanger: A co-design framework for enabling sparse attention using reconfigurable architecture},
  author={Lu, Liqiang and Jin, Yicheng and Bi, Hangrui and Luo, Zizhang and Li, Peng and Wang, Tao and Liang, Yun},
  booktitle={MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture},
  pages={977--991},
  year={2021}
}
```






