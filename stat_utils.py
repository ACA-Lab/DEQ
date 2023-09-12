import torch

def apply_along_axis(function, x, axis: int = -1):
    return torch.stack([function(x_i) for x_i in torch.unbind(x, dim=axis)], dim=axis)

def apply_along_axis_int(function, x, axis: int = -1):
    return torch.stack([torch.tensor(function(x_i), dtype=torch.uint8) for x_i in torch.unbind(x, dim=axis)], dim=axis)

def batch_histogram(data_tensor, num_classes=-1):
    # https://stackoverflow.com/questions/69429586/
    """
    Computes histograms of integral values, even if in batches (as opposed to torch.histc and torch.histogram).
    Arguments:
        data_tensor: a D1 x ... x D_n torch.LongTensor
        num_classes (optional): the number of classes present in data.
                                If not provided, tensor.max() + 1 is used (an error is thrown if tensor is empty).
    Returns:
        A D1 x ... x D_{n-1} x num_classes 'result' torch.LongTensor,
        containing histograms of the last dimension D_n of tensor,
        that is, result[d_1,...,d_{n-1}, c] = number of times c appears in tensor[d_1,...,d_{n-1}].
    """
    # long() is necessary for one_hot to work, or will get RuntimeError: one_hot is only applicable to index tensor
    return torch.nn.functional.one_hot(data_tensor.long(), num_classes).sum(dim=-2)
