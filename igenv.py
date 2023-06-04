from isaacgym import gymtorch

import torch

class IsaacGymEnv:
  def __init__(self, gym, sim, device, env_to_franka, franka_num_dofs):
    self.gym = gym
    self.sim = sim
    self.device = device
    self.num_envs = gym.get_env_count(sim)
    self.env_to_franka = env_to_franka
    self.franka_num_dofs = franka_num_dofs
    
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

    self._step()


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


  def step(self, dof_targets, terms):
    if terms.numel() != 0:
      print("resetting", terms)

    dof_targets[terms] = self.reset_dof_target[terms]
    self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(dof_targets))

    if terms.numel() != 0:
      new_dof_state = torch.zeros_like(self.reset_dof_state)
      new_dof_state[terms, :, 0] = self.reset_dof_target[terms]
      new_dof_state = new_dof_state.view(self.num_envs * self.franka_num_dofs, 2)
      franka_fails = self.env_to_franka[terms]
      franka_fails_int32 = franka_fails.to(device=self.device, dtype=torch.int32)
      self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(new_dof_state), gymtorch.unwrap_tensor(franka_fails_int32), len(franka_fails_int32))

    self._step()

  def start_cam_access(self):
    self.gym.start_access_image_tensors(self.sim)

  def end_cam_access(self):
    self.gym.end_access_image_tensors(self.sim)