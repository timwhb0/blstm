import torch
"""Utility functions for computing dynamic features."""
def comp_dynamic_feature(inputs, DELTAWINDOW, Batch_size, lengths):
    outputs = []
    for i in range(Batch_size):
        tmp = comp_delta(inputs[i,:lengths[i],:], DELTAWINDOW, lengths[i])
        tmp1 = torch.nn.functional.pad(tmp, (0, 0, 0, torch.max(lengths)-lengths[i]), "constant", 0)
        outputs.append(tmp1)
    return torch.stack(outputs)

def comp_delta(feat, N, length):
# """ Compute delta features from a feature vector sequence.
# Args:
# 	feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
# N: For each frame, calculate delta features based on preceding and following N frames.
# Returns:
# 	A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
# """
    feat = torch.cat([feat[0].unsqueeze(0).repeat(N, 1), feat, feat[-1].unsqueeze(0).repeat(N, 1)], 0)
    denom = sum([2*i*i for i in range(1,N+1)])
    dfeat = torch.sum(torch.stack([j*(feat[N+1+j-1:N+1+j+length-1,:]-feat[N+1-j-1:N+1-j+length-1,:]) for j in range(1,N+1)]), 0)/denom
    return dfeat