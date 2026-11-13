# PATCHED vs upstream/ibp.py — upstream: WocaR-RL/WocaR-DQN/SA-DQN (see upstream/ibp.py)
# Changes: CPU-portable worst_action_select, >= bound comparison for eps=0, gather from unmasked worst_q so 1e9 cannot leak into Bellman targets.
# Rev 4: worst_action_select can also return the mean size of the adversarial action set (logging only).
import torch
import torch.nn as nn
import torch.nn.functional as F

def initial_bounds(x0, epsilon):
    '''
    x0 = input, b x c x h x w
    '''
    upper = x0+epsilon
    lower = x0-epsilon
    return upper, lower

def weighted_bound(layer, prev_upper, prev_lower):
    prev_mu = (prev_upper + prev_lower)/2
    prev_r = (prev_upper - prev_lower)/2
    mu = layer(prev_mu)
    if type(layer)==nn.Linear:
        r = F.linear(prev_r, torch.abs(layer.weight))
    elif type(layer)==nn.Conv2d:
        r = F.conv2d(prev_r, torch.abs(layer.weight), stride=layer.stride, padding=layer.padding)
    upper = mu + r
    lower = mu - r
    return upper, lower

def activation_bound(layer, prev_upper, prev_lower):
    upper = layer(prev_upper)
    lower = layer(prev_lower)
    return upper, lower

def network_bounds(model, x0, epsilon):
    '''
    get inteval bound progation upper and lower bounds for the actiavtion of a model
    
    model: a nn.Sequential module
    x0: input, b x input_shape
    epsilon: float, the linf distance bound is calculated over
    '''
    upper, lower = initial_bounds(x0, epsilon)
    for layer in model.modules():
        if type(layer) in (nn.Sequential,):
            pass
        elif type(layer) in (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.MaxPool2d, nn.Flatten):
            upper, lower = activation_bound(layer, upper, lower)
        elif type(layer) in (nn.Conv2d, nn.Linear):
            upper, lower = weighted_bound(layer, upper, lower)
        else:
            print('Unsupported layer:', type(layer))
    return upper, lower

def worst_action_select(worst_q, upper_q, lower_q, return_set_size=False):
    mask = torch.zeros_like(upper_q, dtype=torch.bool)
    for i in range(upper_q.size()[1]):
        upper = upper_q[:, i].view(upper_q.size()[0], 1)
        if_perturb = (upper >= lower_q).all(1)
        mask[:, i] = if_perturb

    empty_rows = ~mask.any(dim=1)
    if empty_rows.any():
        fallback_actions = upper_q.argmax(dim=1, keepdim=True)
        mask.scatter_(1, fallback_actions, mask.gather(1, fallback_actions) | empty_rows.unsqueeze(1))

    masked_worst_q = worst_q.masked_fill(~mask, float('inf'))
    worst_actions = masked_worst_q.min(1)[-1].unsqueeze(1)
    worst_q = worst_q.gather(1, worst_actions).squeeze(1)

    if return_set_size:   # rev 4: mean number of actions the attacker can force (1 = only the greedy action)
        return worst_q, mask.sum(dim=1).float().mean()
    return worst_q






