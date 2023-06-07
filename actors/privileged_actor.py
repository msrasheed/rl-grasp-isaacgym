import torch
import torch.nn as nn
import torch.distributions as tdis

from .util import init

class PrivilegedActor(nn.Module):
  def __init__(self):
    super().__init__()
    self.objE = nn.Sequential(
      init(nn.Linear(7, 256)),
      nn.LayerNorm(256),
      nn.ReLU(inplace=True),
      init(nn.Linear(256, 256))
    )
    self.dofsE = nn.Sequential(
      init(nn.Linear(18, 256)),
      nn.LayerNorm(256),
      nn.ReLU(inplace=True),
      init(nn.Linear(256, 256))
    )
    self.normConcat = nn.LayerNorm(512)
    self.actionHead = nn.Sequential(
      init(nn.Linear(512, 256)),
      nn.LayerNorm(256),
      nn.ReLU(inplace=True),
      nn.Linear(256, 18)
    )
    self.valueHead = nn.Sequential(
      init(nn.Linear(512, 256)),
      nn.LayerNorm(256),
      nn.ReLU(inplace=True),
      nn.Linear(256, 1)
    )
      
  def forward(self, obj_dof, dofs, acts):
    objE = self.objE(obj_dof)
    dofE = self.dofsE(dofs)
    cat = self.normConcat(torch.cat((objE, dofE), dim=1))
    act_stats = self.actionHead(cat)
    value = self.valueHead(cat)
    
    acts_norm = act_stats.view(-1, 2, 9)
    act_dis = tdis.normal.Normal(acts_norm[:, 0, :], torch.abs(acts_norm[:, 1, :]))
    log_acts = act_dis.log_prob(acts).sum(1)
    
    return log_acts, act_dis.entropy(), value
  
  def get_action_value(self, obj_dof, dofs):
    # print(img)
    # print(dofs)
    objE = self.objE(obj_dof)
    dofE = self.dofsE(dofs)
    cat = self.normConcat(torch.cat((objE, dofE), dim=1))
    act_stats = self.actionHead(cat)
    value = self.valueHead(cat)
    
    acts_norm = act_stats.view(-1, 9, 2)
    if torch.any(torch.isnan(acts_norm)):
      print("in pol", obj_dof)
      print("in pol", dofs)
      print("in pol", acts_norm)
      print(torch.any(torch.isinf(obj_dof)))
    act_dis = tdis.normal.Normal(acts_norm[:, :, 0], torch.abs(acts_norm[:, :, 1]))
    acts = act_dis.sample()
    
    return acts, act_dis.log_prob(acts).sum(1), value