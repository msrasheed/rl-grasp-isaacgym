import torch

class RunningMeanStd:
  def __init__(self, shape=[1], device=torch.device('cpu'), epsilon=1e-8):
    self.mean = torch.zeros(*shape, device=device)
    self.var = torch.ones(*shape, device=device)
    self.count = 0
    self.epsilon = epsilon

  def update(self, x):
    batch_mean = torch.mean(x, dim=0)
    batch_var = torch.var(x, dim=0)
    batch_count = x.shape[0]

    delta = batch_mean - self.mean
    tot_count = self.count + batch_count

    new_mean = self.mean + delta * batch_count / tot_count
    m_a = self.var * self.count
    m_b = batch_var * batch_count 
    M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
    new_var = M2 / tot_count

    self.mean = new_mean
    self.var = new_var
    self.count = tot_count

  def normalize(self, obs):
    return (obs - self.mean) / torch.sqrt(self.var + self.epsilon)

  def normalize_no_center(self, obs):
    return obs / torch.sqrt(self.var + self.epsilon)