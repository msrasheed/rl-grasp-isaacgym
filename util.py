import torch

def transform_to_rb_tensor(transform):
  rb_tensor = torch.zeros(13)
  rb_tensor[0] = transform.p.x
  rb_tensor[1] = transform.p.y
  rb_tensor[2] = transform.p.z

  rb_tensor[3] = transform.r.x
  rb_tensor[4] = transform.r.y
  rb_tensor[5] = transform.r.z
  rb_tensor[6] = transform.r.w

  return rb_tensor