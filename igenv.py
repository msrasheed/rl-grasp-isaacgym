from isaacgym import gymtorch

import torch

import running_mean_std as rms

class IsaacGymEnv:
  def __init__(self, gym, sim, franka_num_dofs, target_height,
               device, env_to_box, env_to_franka, box_rb_idx, hand_rb_idx,
               dof_state_tensor, cam_tensors_arr, rb_states):
    self.gym = gym
    self.sim = sim
    self.num_envs = gym.get_env_count(sim)
    self.franka_num_dofs = franka_num_dofs
    self.target_height = target_height
    self.device = device

    self.env_to_box = env_to_box
    self.env_to_franka = env_to_franka
    self.box_rb_idx = box_rb_idx
    self.hand_rb_idx = hand_rb_idx

    self.dof_state_tensor = dof_state_tensor
    self.cam_tensors_arr = cam_tensors_arr
    self.rb_states = rb_states
    
    self.first_reset_called = False

  def reset(self, dof_target_tensor, dof_state_tensor, root_body_tensor):
    print("resetting")
    self.reset_dof_target = dof_target_tensor
    self.reset_dof_state = dof_state_tensor
    self.reset_root_body = root_body_tensor

    if self.first_reset_called:
      self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(dof_target_tensor))
      self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(dof_state_tensor))
      self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_body_tensor))
    else: self.first_reset_called = True

    self.env_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self.rewards = torch.zeros(self.num_envs, device=self.device)
    self._step()
    cam_obs = self.get_cam_obs()
    return (cam_obs, self.dof_state_tensor)


  def _step(self):
    # step the physics
    self.gym.simulate(self.sim)
    self.gym.fetch_results(self.sim, True)

    self.gym.refresh_dof_state_tensor(self.sim)
    self.gym.refresh_rigid_body_state_tensor(self.sim)
    self.gym.refresh_actor_root_state_tensor(self.sim)

    self.gym.step_graphics(self.sim)
    # render sensors and refresh camera tensors
    self.gym.render_all_camera_sensors(self.sim)


  def step(self, dof_targets):
    self.rewards = torch.zeros(self.num_envs, device=self.device)

    terms = torch.nonzero(self.env_reset)
    if terms.numel() != 0:
      print("resetting", terms)

    dof_targets[terms] = self.reset_dof_target[terms]
    self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(dof_targets))

    if terms.numel() != 0:
      franka_fails = self.env_to_franka[terms]
      franka_fails_int32 = franka_fails.to(device=self.device, dtype=torch.int32)
      self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.reset_dof_state), gymtorch.unwrap_tensor(franka_fails_int32), len(franka_fails_int32))

      box_fails = self.env_to_box[terms]
      box_fails_int32 = box_fails.to(device=self.device, dtype=torch.int32)
      box_franka_idx = torch.cat((franka_fails_int32, box_fails_int32))
      self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.reset_root_body), gymtorch.unwrap_tensor(box_franka_idx), len(box_franka_idx))

    self._step()
    cam_obs = self.get_cam_obs()
    self.calc_results()

    ret_env_reset = self.env_reset
    self.env_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    return (cam_obs, self.dof_state_tensor), self.rewards, ret_env_reset


  def start_cam_access(self):
    self.gym.start_access_image_tensors(self.sim)

  def end_cam_access(self):
    self.gym.end_access_image_tensors(self.sim)


  def get_cam_obs(self):
    self.gym.start_access_image_tensors(self.sim)
    cam_tensors = torch.stack(self.cam_tensors_arr)
    infs_idx = torch.isinf(cam_tensors)
    cam_with_infs = torch.unique(torch.nonzero(infs_idx)[:, 0])
    cam_tensors[infs_idx] = 0
    self.gym.end_access_image_tensors(self.sim)

    self.rewards[cam_with_infs] -= 1
    
    return cam_tensors

  def calc_results(self):
    hand_dist = torch.norm(self.rb_states[self.box_rb_idx, 0:3] \
                           -self.rb_states[self.hand_rb_idx, 0:3], dim=1)
    box_height = self.target_height - self.rb_states[self.box_rb_idx, 2] 
    finished = box_height > self.target_height

    self.env_reset |= finished

    self.rewards = -(hand_dist + box_height)

  
class NormalizeWrapper:
  def __init__(self, env, obs_shapes, gamma, epsilon=1e-8):
    self.venv = env
    self.num_envs = env.num_envs
    self.device = env.device
    self.ret = torch.zeros(env.num_envs, device=env.device)
    self.prev_reset = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    self.obs_rms = [rms.RunningMeanStd(shape, self.device, epsilon) for shape in obs_shapes]
    self.rwd_rms = rms.RunningMeanStd(device=env.device, epsilon=epsilon)
    self.gamma = gamma
    
  def reset(self, dof_target_tensor, dof_state_tensor, root_body_tensor):
    self.ret = torch.zeros(self.venv.num_envs, device=self.venv.device)
    obs = self.venv.reset(dof_target_tensor, dof_state_tensor, root_body_tensor)
    return self._obfilt(obs)

  def _obfilt(self, obs):
    [r.update(ob) for r, ob in zip(self.obs_rms, obs)]
    return [r.normalize(ob) for r, ob in zip(self.obs_rms, obs)]

  def step(self, dof_targets):
    obs, rewds, last_step = self.venv.step(dof_targets)
    normobs = self._obfilt(obs)
    self.ret = self.ret * self.gamma + rewds
    self.rwd_rms.update(self.ret)

    normrwds = self.rwd_rms.normalize_no_center(rewds)
    # normrwds = rewds
    self.ret[self.prev_reset] = 0
    self.prev_reset = last_step
    return normobs, normrwds, last_step