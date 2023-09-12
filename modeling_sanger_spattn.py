import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from stat_utils import apply_along_axis_int, batch_histogram, apply_along_axis

import time


'''
LOG_LOAD_BALANCE = os.getenv('LOG_LOAD_BALANCE', False)

if LOG_LOAD_BALANCE:
    cur_time = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
    csv_path = Path(cur_time + r'load_balance.csv')
    assert not csv_path.exists(), f'{csv_path} already exists.'
    csv_file = csv_path.open('w')
    csv_file.write('50%-no-skip,50%-skip,25%-no-skip,25%-skip,overall-sparsity\n')
'''

import envvar_utils
def init_ratio_csv():
    if envvar_utils.log_lb_enabled() and envvar_utils.is_in_eval():
        global csv_file
        cur_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        if os.environ['TASK_NAME'] is None:
            csv_path = Path(cur_time + r'_sanger.csv')
        else:
            csv_path = Path(cur_time + r'_sanger_' + os.environ['TASK_NAME'] + r'.csv')
        assert not csv_path.exists(), f'{csv_path} already exists.'
        csv_file = csv_path.open('w')
        csv_file.write('attn_mask_ratio, overall-sparsity\n')


'''
attn_mask中生效的位置，即非padding位置，占总位置的比例
'''
def _eval_attn_mask(attn_mask):
    # attn_mask: bool, [batch_size, 1, seq_len, seq_len]

    # attn_mask is used to mask out padding tokens, is prior to refienment_mask
    return attn_mask.mean().item()


def _eval_load_balance(sparsity_mask, attn_mask, num_ports=64, num_pes=16, no_skip=False):
    # sparsity_mask: bool, [batch_size, num_heads, seq_len, seq_len]
    # attn_mask: bool, [batch_size, 1, seq_len, seq_len]
    batch_size, num_heads, seq_len, seq_len = sparsity_mask.shape
    assert seq_len % num_ports == 0

    # split sparsity mask into `num_ports`-dim vectors
    # sparsity_mask: [batch_size, num_heads, seq_len * seq_len / num_ports, num_ports]
    sparsity_mask = sparsity_mask.view(batch_size, num_heads, -1, num_ports)

    # count nonzeros in each vector
    # num_nonzero: [batch_size, num_heads, seq_len * seq_len / num_ports]
    num_nonzero = sparsity_mask.sum(dim=-1)
    
    # split attention mask into `num_ports`-dim vectors
    # attn_mask: [batch_size, 1, seq_len * seq_len / num_ports, num_ports]
    attn_mask = attn_mask.view(batch_size, 1, -1, num_ports)

    # vector-wise attention mask: mask out vectors that are completely covered by the original attention mask 
    # attn_mask: bool, [batch_size, 1, seq_len * seq_len / num_ports]
    attn_mask = attn_mask.sum(dim=-1).ne(0)
    
    # filter out masked vectors from num_nonzero
    # num_nonzero: 1-D vector
    num_nonzero = torch.masked_select(num_nonzero, attn_mask)
    
    # count and skip all-zero vectors
    skip_mask = num_nonzero.ne(0)
    num_skips = skip_mask.sum()

    # filter out skipped vectors from num_nonzero
    num_nonzero = torch.masked_select(num_nonzero, skip_mask)
    
    # split non-empty vectors into segments with nnz no greater than num_pes
    # assuming num_pes = 3, a vector of length 10 can be divided into four segments [3, 3, 3, 1]
    # in this case, there are three full segments (where all pes are occupied) and one unfull remnant
    num_splits = num_nonzero / num_pes
    num_full_splits = num_splits.floor().sum()
    num_all_splits = num_splits.ceil().sum()
    
    # a full segment leads to a pe utilization of 100%
    # while pe util of a remnant segment is calculated as num-occupied-pes / num-pes
    acc_full_split_utils = num_full_splits * 1.0
    acc_remn_split_utils = num_splits.frac().sum()
    # accumulated pe utilization of all segments
    acc_all_split_utils = acc_full_split_utils + acc_remn_split_utils

    if no_skip:
        pe_util = acc_all_split_utils / (num_all_splits + num_skips)
    else:
        pe_util = acc_all_split_utils / num_all_splits

    return pe_util.item()


def _eval_overall_sparsity(sparsity_mask, attn_mask):
    # sparsity_mask: bool, [batch_size, num_heads, seq_len, seq_len]
    # attn_mask: bool, [batch_size, 1, seq_len, seq_len]
    scaling_factor = attn_mask.mean(dim=(1, 2, 3))
    sparsity_per_seq = (sparsity_mask * attn_mask).mean(dim=(1, 2, 3))
    overall_sparsity = (sparsity_per_seq / scaling_factor).mean().item()
    return overall_sparsity


def gen_sparsity_mask(threshold, attention_scores, attn_mask):
    print(attention_scores.shape)
    print("before===attention_scores:   ",attention_scores)
    attention_scores = F.softmax(attention_scores+attn_mask, dim=-1)
    print("after===attention_scores:   ",attention_scores)
    sparsity_mask = attention_scores > threshold
    sparsity_mask = sparsity_mask.type_as(attention_scores)

    '''
    if LOG_LOAD_BALANCE and random.random() < 3e-2:
        attn_mask = (attn_mask > -1).float()
        attn_mask = attn_mask * attn_mask.permute(0, 1, 3, 2)
        logs = [
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=32, no_skip=True), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=32, no_skip=False), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=16, no_skip=True), 
            _eval_load_balance(sparsity_mask, attn_mask, num_pes=16, no_skip=False),
            _eval_overall_sparsity(sparsity_mask, attn_mask)
        ]
        csv_file.write(','.join([f'{stat:.6f}' for stat in logs]) + '\n')
    '''
    if envvar_utils.log_lb_enabled() and envvar_utils.is_in_eval() :
        attn_mask = (attn_mask > -1).float()
        attn_mask = attn_mask * attn_mask.permute(0, 1, 3, 2)
        logs = [
            _eval_attn_mask(attn_mask),
            _eval_overall_sparsity(sparsity_mask, attn_mask)
        ]
        try:
            csv_file
        except NameError:
            init_ratio_csv()
        csv_file.write(','.join([f'{stat:.6f}' for stat in logs]) + '\n')
    
    sparsity_mask = (1.0 - sparsity_mask) * -10000.0
    return sparsity_mask.detach()

import numpy as np
import cupy as cp
from numba import jit, prange
from typing import Union

#@jit(nopython=True, parallel=True)
def get_otsu_threshold(hist: Union[torch.Tensor, np.ndarray, cp.ndarray]) -> int:
    if hist.shape != (256,):
        raise ValueError('histogram must be of shape (256,), you have {}', hist.shape)
    if type(hist) == torch.Tensor:
        sum_val = torch.sum(torch.arange(256, device=hist.device) * hist)
    elif type(hist) == np.ndarray:
        sum_val = np.sum(np.arange(256) * hist)
    elif type(hist) == cp.ndarray:
        sum_val = cp.sum(cp.arange(256) * hist)
    sum_b = 0
    w_b = 0
    w_f = hist.sum()
    max_var = 0
    threshold = 0

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue

        w_f -= hist[t]
        if w_f == 0:
            break

        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_val - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2

        if var > max_var:
            max_var = var
            threshold = t
    
    return threshold

''' inefficient implementation
def get_otsu_threshold_wrapper(scaled_scores, granularity="batch"):
    assert len(scaled_scores.shape) == 4, f"Scaled scores should be [batch_size, num_heads, seq_len, seq_len], got {scaled_scores.shape}"
    if granularity == "batch":
        thresholds_ = torch.zeros(scaled_scores.shape[0], dtype=torch.uint8, device=scaled_scores.device)
    elif granularity == "head":
        thresholds_ = torch.zeros(scaled_scores.shape[0], scaled_scores.shape[1],
                                  dtype=torch.uint8, device=scaled_scores.device)
    elif granularity == "sequence":
        thresholds_ = torch.zeros(scaled_scores.shape[0], scaled_scores.shape[1],
                                  scaled_scores.shape[2], dtype=torch.uint8, device=scaled_scores.device)
    else:
        raise ValueError(f"granularity must be one of 'batch', 'head', or 'sequence'")
    
    if granularity == "batch":
        scores = scaled_scores.view(scaled_scores.shape[0], -1)
        hist = batch_histogram(scores, num_classes=256)
        thresholds = apply_along_axis_int(get_otsu_threshold, hist.view(-1, 256), axis=0)
    elif granularity == "head":
        scores = scaled_scores.view(scaled_scores.shape[0], scaled_scores.shape[1], -1)
        hist = batch_histogram(scores, num_classes=256)
        print("hist shape", hist.shape)
        assert hist.shape[-1]==256
        thresholds = apply_along_axis_int(get_otsu_threshold, hist.view(-1, 256), axis=0)
        thresholds = thresholds.view(scaled_scores.shape[0], scaled_scores.shape[1])
    elif granularity == "sequence":
        scores = scaled_scores.view(scaled_scores.shape[0], scaled_scores.shape[1], scaled_scores.shape[2], -1)
        hist = batch_histogram(scores, num_classes=256)
        thresholds = apply_along_axis_int(get_otsu_threshold, hist.view(-1, 256), axis=0)
        thresholds = thresholds.view(scaled_scores.shape[0], scaled_scores.shape[1], scaled_scores.shape[2])
    assert thresholds.shape == thresholds_.shape, f"Thresholds should be {thresholds_.shape}, got {thresholds.shape}"
    return thresholds
'''

import concurrent.futures
@jit(nopython=False, parallel=True)
def get_otsu_threshold_wrapper(scaled_scores, granularity="head"):
    #assert len(scaled_scores.shape) == 4, f"Scaled scores should be [batch_size, num_heads, seq_len, seq_len], got {scaled_scores.shape}"
    if granularity == "batch":
        thresholds = torch.zeros(scaled_scores.shape[0], dtype=torch.uint8, device=scaled_scores.device)
    elif granularity == "head":
        thresholds = torch.zeros(scaled_scores.shape[0], scaled_scores.shape[1],
                                  dtype=torch.uint8, device=scaled_scores.device)
    elif granularity == "sequence":
        thresholds = torch.zeros(scaled_scores.shape[0], scaled_scores.shape[1],
                                  scaled_scores.shape[2], dtype=torch.uint8, device=scaled_scores.device)
    #else:
    #    raise ValueError(f"granularity must be one of 'head', 'sequence', or 'row'")
    
    scaled_scores = scaled_scores.cpu().detach().numpy()
    #scaled_scores = cp.asarray(scaled_scores)
    # TODO: improve efficiency
    if granularity == "batch":
        for i in prange(scaled_scores.shape[0]): # batch idx
            #hist = torch.histc(scaled_scores[i].view(-1), bins=256, min=0, max=255)
            #hist = torch.bincount(scaled_scores[i].view(-1).cpu(), minlength=256)
            hist = np.bincount(scaled_scores[i].ravel(), minlength=256)
            #hist, bins = np.histogram(scaled_scores[i].ravel(), bins=256, range=(0, 255))
            threshold = get_otsu_threshold(hist)
            thresholds[i] = threshold
    elif granularity == "head":
        for i in prange(scaled_scores.shape[0]): # batch idx
            for j in prange(scaled_scores.shape[1]): # head idx
                hist = np.bincount(scaled_scores[i][j].ravel(), minlength=256)
                threshold = get_otsu_threshold(hist)
                thresholds[i][j] = threshold
    elif granularity == "sequence":
        for i in prange(scaled_scores.shape[0]): # batch idx
            for j in prange(scaled_scores.shape[1]): # head idx
                for k in prange(scaled_scores.shape[2]): # seq idx
                    hist = np.bincount(scaled_scores[i][j][k].ravel(), minlength=256)
                    threshold = get_otsu_threshold(hist)
                    thresholds[i][j][k] = threshold

    return thresholds


def gen_sparsity_mask_otsu(attention_scores, attn_mask, granularity="head"):
    attention_scores = F.softmax(attention_scores+attn_mask, dim=-1)
    # 
    # 1. scale attention_scores to uint8 within [0, 255], apply it row-wise, i.e. dim=-1
    # 2. apply Otsu's adaptive thresholding algorithm to attention_scores to get thresholds row-wise, i.e. dim=-1
    # 3. generate sparsity_mask according to thresholds, which is to say, for each row in attention_scores, 
    # all positions whose attention_scores >= threshold of this row should be assigned a sparsity_mask value of 1.0
    # sparsity_mask should be of the same shape as attention_scores.
    
    # Scale attention_scores to uint8 within [0, 255], apply it row-wise
    scaled_scores = (attention_scores * 255).type(torch.uint8)

    thresholds = get_otsu_threshold_wrapper(scaled_scores, granularity)

    # Generate sparsity_mask according to thresholds
    sparsity_mask = torch.zeros_like(attention_scores)
    # TODO: improve efficiency
    if granularity == "batch":
        for i in range(thresholds.shape[0]):
            mask = scaled_scores[i] >= thresholds[i]
            sparsity_mask[i] = mask.float()
    elif granularity == "head":
        for i in range(thresholds.shape[0]):
            for j in range(thresholds.shape[1]):
                mask = scaled_scores[i][j] >= thresholds[i][j]
                sparsity_mask[i][j] = mask.float()
    elif granularity == "sequence":
        for i in range(thresholds.shape[0]):
            for j in range(thresholds.shape[1]):
                for k in range(thresholds.shape[2]):
                    mask = scaled_scores[i][j][k] >= thresholds[i][j][k]
                    sparsity_mask[i][j][k] = mask.float()
    else:
        raise ValueError(f"granularity must be one of 'batch', 'head', or 'sequence'")

    sparsity_mask = sparsity_mask.type_as(attention_scores)
    
    # TODO: log sparsity
    
    sparsity_mask = (1.0 - sparsity_mask) * -10000.0

    return sparsity_mask.detach()


def quant_qk_matmul(query_layer, key_layer, config, quant_matmul=None):
    assert getattr(config, 'quant_qk', False)
    do_normalize = getattr(config, 'normalize_qk', False)
    if do_normalize:
        assert config.normalize_qk == 'inner_product'
        query_norm = query_layer.norm(dim=-1, keepdim=True)
        key_norm = key_layer.norm(dim=-2, keepdim=True)
        normed_query_layer = query_layer / query_norm
        normed_key_layer = key_layer / key_norm
        quant_attention_scores = quant_matmul(normed_query_layer, normed_key_layer)
        quant_attention_scores *= query_norm * key_norm
    else:
        quant_attention_scores = quant_matmul(query_layer, key_layer)
    return quant_attention_scores

# attn_scores:  [batch_size, num_attention_heads, seq_len, seq_len] e.g. [32, 12, 128, 128]
# attn_mask:    [batch_size, 1, seq_len, seq_len]
def prune_attn_scores(attn_scores, attn_mask, config):
    assert getattr(config, 'prune_score', False)
    method = config.prune_score['method']
    if method == 'probablity_threshold':
        threshold = config.prune_score['threshold']
        sparsity_mask = gen_sparsity_mask(threshold, attn_scores, attn_mask)
    else:
        raise NotImplementedError(f'Pruning method {method} is not implemented.')
    return sparsity_mask


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
