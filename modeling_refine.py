import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F

import envvar_utils
import time

def init_row_ratio_csv():
    if envvar_utils.log_row_ratio_enabled() and envvar_utils.is_in_eval():
        global row_csv_file
        cur_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        if os.environ['TASK_NAME'] is None:
            csv_path = Path(cur_time + r'_rowwise_refine.csv')
        else:
            csv_path = Path(cur_time + r'_rowwise_refine_' + os.environ['TASK_NAME'] + r'.csv')
        assert not csv_path.exists(), f'{csv_path} already exists.'
        row_csv_file = csv_path.open('w')
        row_csv_file.write('batch_idx, head_idx, seq_len, masked_seq_len, row_idx, n_refine, refinement_ratio\n')


def init_ratio_csv():
    if envvar_utils.log_lb_enabled() and envvar_utils.is_in_eval():
        global csv_file
        cur_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        if os.environ['TASK_NAME'] is None:
            csv_path = Path(cur_time + r'_refine.csv')
        else:
            csv_path = Path(cur_time + r'_refine_' + os.environ['TASK_NAME'] + r'.csv')
        assert not csv_path.exists(), f'{csv_path} already exists.'
        csv_file = csv_path.open('w')
        csv_file.write('attn_mask_ratio, refinement_ratio\n')

def _eval_attn_mask(attn_mask):
    # attn_mask: bool, [batch_size, 1, seq_len, seq_len]

    # attn_mask is used to mask out padding tokens, is prior to refienment_mask
    return attn_mask.mean().item()


def _eval_refinement(refienment_mask, attn_mask):
    # refienment_mask: bool, [batch_size, num_heads, seq_len, seq_len]
    # attn_mask: bool, [batch_size, 1, seq_len, seq_len]

    # attn_mask is used to mask out padding tokens, is prior to refienment_mask
    scaling_factor = attn_mask.mean(dim=(1, 2, 3))  # [batch_size]
    refinement_per_seq = (refienment_mask * attn_mask).mean(dim=(1, 2, 3))
    overall_refinement = (refinement_per_seq / scaling_factor).mean().item()
    return overall_refinement


def gen_refinement_mask(threshold, attention_scores, attn_mask, inv=False):
    attention_scores = F.softmax(attention_scores+attn_mask, dim=-1)
    refinement_mask = attention_scores > threshold if not inv else attention_scores <= threshold
    refinement_mask = refinement_mask.type_as(attention_scores)

    # log refinement ratio over all input batches
    # then you can average it manually
    if envvar_utils.log_lb_enabled() and envvar_utils.is_in_eval():  # and random.random() < 3e-2:
        attn_mask = (attn_mask > -1).float()
        attn_mask = attn_mask * attn_mask.permute(0, 1, 3, 2)
        logs = [
            _eval_attn_mask(attn_mask),
            _eval_refinement(refinement_mask, attn_mask)
        ]
        try:
            csv_file
        except NameError:
            init_ratio_csv()
        csv_file.write(','.join([f'{stat:.6f}' for stat in logs]) + '\n')

        if envvar_utils.log_row_ratio_enabled():
            raise NotImplementedError('log_row_ratio_enabled is not implemented for gen_refinement_mask yet.')  # TODO implement later

    #sparsity_mask = (1.0 - sparsity_mask) * -10000.0
    return refinement_mask.detach()


def gen_refinement_mask_adamax(max_pct, attention_scores, attn_mask, granularity='head', inv=False, before_softmax=False):
    # attention_scores: [batch_size, num_heads, seq_len, seq_len]
    if not before_softmax:
        attention_scores = F.softmax(attention_scores+attn_mask, dim=-1)
    
    b, h, s, _ = attention_scores.size()

    refinement_mask = None

    if granularity == 'batch':
        # Compute the max attention score within each batch
        max_scores, _ = torch.max(attention_scores.view(b, -1), dim=1)
        refinement_mask = attention_scores > max_pct * max_scores[..., None, None, None] if not inv \
            else attention_scores <= max_pct * max_scores[..., None, None, None]
    elif granularity == 'head':
        # Compute the max attention score per head
        max_scores, _ = torch.max(attention_scores.view(b, h, -1), dim=2)
        refinement_mask = attention_scores > max_pct * max_scores[..., None, None] if not inv \
            else attention_scores <= max_pct * max_scores[..., None, None]
    elif granularity == 'seq':
        # Compute the max attention score within each head and sequence position
        max_scores, _ = torch.max(attention_scores, dim=3)
        refinement_mask = attention_scores > max_pct * max_scores[..., None] if not inv \
            else attention_scores <= max_pct * max_scores[..., None]
    else:
        raise ValueError(f'Invalid granularity: {granularity}')
    
    refinement_mask = refinement_mask.type_as(attention_scores)

    if before_softmax:  # firstly generate refine mask
        attention_scores = F.softmax(attention_scores+attn_mask, dim=-1)

    # log refinement ratio over all input batches
    # then you can average it manually
    if envvar_utils.log_lb_enabled() and envvar_utils.is_in_eval():  # and random.random() < 3e-2:
        attn_mask = (attn_mask > -1).float()
        attn_mask = attn_mask * attn_mask.permute(0, 1, 3, 2)
        logs = [
            _eval_attn_mask(attn_mask),
            _eval_refinement(refinement_mask, attn_mask)
        ]
        try:
            csv_file
        except NameError:
            init_ratio_csv()
        csv_file.write(','.join([f'{stat:.6f}' for stat in logs]) + '\n')

        ## TODO: this is for logging row-wise refinement ratio
        if envvar_utils.log_row_ratio_enabled():
            import math
            for b_idx in range(b):
                for h_idx in range(h):
                    masked_seq_len = int(math.sqrt(attn_mask[b_idx].sum())) #TODO
                    assert masked_seq_len ** 2 == attn_mask[b_idx].sum(), f"not square: {attn_mask[b_idx].sum()}"
                    for row_idx in range(masked_seq_len):
                        n_refine = refinement_mask[b_idx, h_idx, row_idx].sum()
                        row_logs = [
                            b_idx, # batch_idx
                            h_idx, # head_idx
                            s, # seq_len
                            masked_seq_len, # masked_seq_len
                            row_idx, # row_idx
                            n_refine, # n_refine
                            n_refine / masked_seq_len # row_refinement_ratio
                        ]
                        try:
                            row_csv_file
                        except NameError:
                            init_row_ratio_csv()
                        row_csv_file.write(','.join([f'{stat}' for stat in row_logs]) + '\n')


    #sparsity_mask = (1.0 - sparsity_mask) * -10000.0
    return refinement_mask.detach()


def quant_qk_matmul(query_layer, key_layer, config, quant_matmul=None):
    assert getattr(config, 'quant_qk', False)
    do_normalize = getattr(config, 'normalize_qk', False)
    if do_normalize:
        assert config.normalize_qk == 'inner_product'
        query_norm = query_layer.norm(dim=-1, keepdim=True)
        key_norm = key_layer.norm(dim=-2, keepdim=True)
        normed_query_layer = query_layer / query_norm
        normed_key_layer = key_layer / key_norm
        quant_attention_scores = quant_matmul(
            normed_query_layer, normed_key_layer)
        quant_attention_scores *= query_norm * key_norm
    else:
        quant_attention_scores = quant_matmul(query_layer, key_layer)
    return quant_attention_scores

# attn_scores:  [batch_size, num_attention_heads, seq_len, seq_len] e.g. [32, 12, 128, 128]
# attn_mask:    [batch_size, 1, seq_len, seq_len]


def refine_attn_scores(attn_scores, attn_mask, config):
    assert getattr(config, 'refine_score', False)
    method = config.refine_score['method']
    if method == 'probablity_threshold':
        threshold = config.refine_score['threshold']
        refinement_mask = gen_refinement_mask(
            threshold, attn_scores, attn_mask)
    elif method == 'inv_probablity_threshold':
        threshold = config.refine_score['threshold']
        refinement_mask = gen_refinement_mask(
            threshold, attn_scores, attn_mask, inv=True)
    elif method == 'adaptive_maximum_threshold':
        max_pct = config.refine_score['max_pct']
        granuality = config.refine_score['granuality']
        before_softmax = config.refine_score.get('before_softmax', False)
        if before_softmax is None:
            before_softmax = False
        refinement_mask = gen_refinement_mask_adamax(
            max_pct, attn_scores, attn_mask, granularity=granuality, before_softmax=before_softmax)
    elif method == 'inv_adaptive_maximum_threshold':
        max_pct = config.refine_score['max_pct']
        granuality = config.refine_score['granuality']
        refinement_mask = gen_refinement_mask_adamax(
            max_pct, attn_scores, attn_mask, granularity=granuality, inv=True)
    elif method == 'otsu_threshold':
        raise NotImplementedError('Otsu thresholding is not implemented.')
    else:
        raise NotImplementedError(
            f'Refinement method {method} is not implemented.')
    return refinement_mask


# def sanger_sparse_attention(query_layer, key_layer, attention_mask, config, quant_matmul=None):
#     # query_layer:    [batch_size, num_attention_heads, seq_len, attention_head_size]
#     # key_layer:      [batch_size, num_attention_heads, attention_head_size, seq_len]
#     # attention_mask: [batch_size, num_attention_heads, seq_len, seq_len]

#     do_quant = getattr(config, 'quant_qk', False)
#     do_prune = getattr(config, 'prune_score', False)

#     attention_head_size = query_layer.shape[-1]
#     scale_factor = math.sqrt(attention_head_size)

#     attention_scores = torch.matmul(query_layer, key_layer)
#     attention_scores = attention_scores / scale_factor

#     if do_quant:
#         quant_attention_scores = quant_qk_matmul(query_layer, key_layer, config, quant_matmul)
#         quant_attention_scores = quant_attention_scores / scale_factor
#     else:
#         quant_attention_scores = None

#     if do_prune:
#         attn_scores = quant_attention_scores if do_quant else attention_scores
#         sparsity_mask = prune_attn_scores(attn_scores, attention_mask, config)
#         attention_scores += sparsity_mask

#     attention_scores = attention_scores + attention_mask
#     attention_probs = F.softmax(attention_scores, dim=-1)

#     return attention_probs
