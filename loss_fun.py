import torch
import torch.nn.functional as F

def rank_loss(pred, true):
    ones = torch.ones(1, pred.size(0), device=pred.device, dtype=pred.dtype)

    pred_expand = pred.matmul(ones)
    pred_diff = pred_expand - pred_expand.transpose(0, 1)

    true_expand = true.matmul(ones)
    true_diff = true_expand - true_expand.transpose(0, 1)

    loss = torch.mean(F.relu(- pred_diff * true_diff))
    return loss


