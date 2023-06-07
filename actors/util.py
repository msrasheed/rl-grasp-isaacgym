import torch.nn as nn

def init(m, gain=0.01, activate=True):

  def helper(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
      bias_init(module.bias.data)
    return module

  if activate:
    if isinstance(m, nn.Conv2d):
      gain = nn.init.calculate_gain('conv2d')
    else:
      gain = nn.init.calculate_gain('relu')
  return helper(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)