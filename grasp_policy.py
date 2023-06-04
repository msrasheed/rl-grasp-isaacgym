import torch
import torch.nn as nn
import torch.distributions as tdis

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


class Policy(nn.Module):
  def __init__(self):
    super().__init__()
    self.imgE = nn.Sequential(
      init(nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)),
      nn.LayerNorm([64,64]),
      nn.ReLU(inplace=True),
      init(nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)),
      nn.LayerNorm([32,32]),
      nn.ReLU(inplace=True),
      init(nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)),
      nn.LayerNorm([16,16]),
      nn.ReLU(inplace=True),
      nn.Flatten(),
      init(nn.Linear(4096, 512))
    )
    self.dofsE = nn.Sequential(
      init(nn.Linear(18, 256)),
      nn.LayerNorm(256),
      nn.ReLU(inplace=True),
      init(nn.Linear(256, 512))
    )
    self.normConcat = nn.LayerNorm(1024)
    self.actionHead = nn.Sequential(
      init(nn.Linear(1024, 256)),
      nn.LayerNorm(256),
      nn.ReLU(inplace=True),
      nn.Linear(256, 18)
    )
    self.valueHead = nn.Sequential(
      init(nn.Linear(1024, 256)),
      nn.LayerNorm(256),
      nn.ReLU(inplace=True),
      nn.Linear(256, 1)
    )
      
  def forward(self, img, dofs, acts):
    imgE = self.imgE(img)
    dofE = self.dofsE(dofs)
    cat = self.normConcat(torch.cat((imgE, dofE), dim=1))
    act_stats = self.actionHead(cat)
    value = self.valueHead(cat)
    
    acts_norm = act_stats.view(-1, 2, 9)
    act_dis = tdis.normal.Normal(acts_norm[:, 0, :], torch.abs(acts_norm[:, 1, :]))
    log_acts = act_dis.log_prob(acts).sum(1)
    
    return log_acts, act_dis.entropy(), value
  
  def get_action_value(self, img, dofs):
    # print(img)
    # print(dofs)
    imgE = self.imgE(img)
    dofE = self.dofsE(dofs)
    cat = self.normConcat(torch.cat((imgE, dofE), dim=1))
    act_stats = self.actionHead(cat)
    value = self.valueHead(cat)
    
    acts_norm = act_stats.view(-1, 9, 2)
    if torch.any(torch.isnan(acts_norm)):
      print("in pol", img)
      print("in pol", dofs)
      print("in pol", acts_norm)
      print(torch.any(torch.isinf(img)))
    act_dis = tdis.normal.Normal(acts_norm[:, :, 0], torch.abs(acts_norm[:, :, 1]))
    acts = act_dis.sample()
    
    return acts, act_dis.log_prob(acts).sum(1), value