from isaacgym import gymapi
from isaacgym import torch_utils
from isaacgym import gymtorch

import torch
import numpy as np
import collections

import util


class PrivelegedBoxEnv:
  def __init__(self, gym, sim, num_envs, device, no_visual):
    self.gym = gym
    self.sim = sim
    self.num_envs = num_envs
    self.device = device
    self.no_visual = no_visual
    
    self.first_reset_called = False

  def create(self):
    self.envs = list()
    self.actor_handles = collections.defaultdict(list)
    self.actor_idx = collections.defaultdict(list)


    # Creating Ground Plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up, if y-up (0,1,0)
    plane_params.distance = 0
    plane_params.static_friction = 1
    plane_params.dynamic_friction = 1
    plane_params.restitution = 0

    self.gym.add_ground(self.sim, plane_params)

    # load franka asset
    asset_root = "../isaacgym/assets"
    asset_file = "urdf/franka_description/robots/franka_panda.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.01
    asset_options.fix_base_link = True
    # asset_options.disable_gravity = True
    asset_options.flip_visual_attachments = True
    franka_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    # franka pose, for creating actor
    franka_pose = gymapi.Transform()
    franka_pose.p = gymapi.Vec3(0,0,0)

    # get franka dof props and joint ranges
    franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
    franka_lower_limits = franka_dof_props["lower"]
    franka_upper_limits = franka_dof_props["upper"]
    franka_ranges = franka_upper_limits - franka_lower_limits
    franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

    # set up drives
    franka_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    franka_dof_props["damping"].fill(40.0)
    franka_dof_props["stiffness"][:7].fill(400.0)
    franka_dof_props["stiffness"][7:].fill(800.0)

    # default dof states and position targets
    franka_num_dofs = self.gym.get_asset_dof_count(franka_asset)
    default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
    default_dof_pos[:7] = franka_mids[:7]
    # grippers open
    default_dof_pos[7:] = franka_upper_limits[7:]

    # setup default dof pose
    default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
    default_dof_state["pos"] = default_dof_pos
    default_dof_pos_tensor = torch_utils.to_torch(default_dof_pos, device=self.device)


    # create table asset
    table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

    # table pose
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

    self.target_height = table_dims.z + .2


    # create object asset
    box_size = 0.045
    box_asset = self.gym.create_box(self.sim, box_size, box_size, box_size)
    box_pose = gymapi.Transform()

    # configure env grid
    num_envs = self.num_envs
    num_per_row = int(np.sqrt(num_envs))
    spacing = 1.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    init_root_body_pose = list()

    for i in range(num_envs):
      env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
      self.envs.append(env)

      # add franka
      franka_handle = self.gym.create_actor(env, franka_asset, franka_pose, "franka", i, 1)
      self.actor_handles["franka"].append(franka_handle)
      # hand_rb_index.append(self.gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM))
      self.actor_idx["franka"].append(self.gym.get_actor_index(env, franka_handle, gymapi.DOMAIN_SIM))
      init_root_body_pose.append(util.transform_to_rb_tensor(franka_pose))

      # set dof props (the drives)
      self.gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)
      # set dof states (sets position, but doesn't stay if targets don't change)
      self.gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)
      # set dof targets (otherwise robot just stands up to default targets)
      self.gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

      # add table
      table_handle = self.gym.create_actor(env, table_asset, table_pose, "table", i, 0)
      self.actor_handles["table"].append(table_handle)
      self.actor_idx["table"].append(self.gym.get_actor_index(env, table_handle, gymapi.DOMAIN_SIM))
      init_root_body_pose.append(util.transform_to_rb_tensor(table_pose))

      # add box - random position/rotation on table
      box_pose.p.x = table_pose.p.x # + np.random.uniform(-0.2, 0.1)
      box_pose.p.y = table_pose.p.y # + np.random.uniform(-0.3, 0.3)
      box_pose.p.z = table_dims.z + 0.5 * box_size
      box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-np.pi, np.pi))
      box_handle = self.gym.create_actor(env, box_asset, box_pose, "box", i, 0)
      self.actor_handles["box"].append(box_handle)
      color = gymapi.Vec3(*np.random.uniform(0, 1, 3))
      self.gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
      # box_rb_index.append(self.gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM))
      self.actor_idx["box"].append(self.gym.get_actor_index(env, box_handle, gymapi.DOMAIN_SIM))
      init_root_body_pose.append(util.transform_to_rb_tensor(box_pose))


    # prepare internal data structures for tensor API
    self.gym.prepare_sim(self.sim)

    # collect dof tensors (frankas dof, other objects don't have dofs)
    _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
    dof_states = gymtorch.wrap_tensor(_dof_states)
    dof_state_reset = dof_states.view(num_envs, franka_num_dofs, 2)
    self.dof_states = dof_states.view(num_envs, franka_num_dofs*2)

    # collect rb tensor for reward calculations later
    _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
    self.rb_states = gymtorch.wrap_tensor(_rb_states)

    # collect root state tensor for resets
    _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
    self.root_bodies = gymtorch.wrap_tensor(_root_tensor)

    # setup reset vals
    reset_dof_state_tensor = torch.zeros_like(dof_state_reset)
    reset_dof_state_tensor[:, :, 0] = default_dof_pos_tensor
    reset_dof_target_tensor = reset_dof_state_tensor[:, :, 0].clone()
    reset_root_body_tensor = torch.stack(init_root_body_pose).to(self.device)

    self.reset_dof_state_tensor = reset_dof_state_tensor
    self.reset_dof_target_tensor = reset_dof_target_tensor
    self.reset_root_body_tensor = reset_root_body_tensor


    self.num_dofs = franka_num_dofs
    self.env_to_dof_actors = self.get_env_to_dof_actor()
    self.env_to_actors = self.get_env_to_actors()
    self.lfing_rb_idx = self.find_actor_rb_index("franka", "panda_leftfinger")
    self.rfing_rb_idx = self.find_actor_rb_index("franka", "panda_rightfinger")
    self.box_rb_idx = self.get_actor_rb_index("box", 0)


  def get_env_to_dof_actor(self):
    return torch.tensor(self.actor_idx["franka"])

  def get_env_to_actors(self):
    return torch.tensor(list(self.actor_idx.values())).transpose(1, 0)

  def find_actor_rb_index(self, actor_name, rb_name, domain=gymapi.DOMAIN_SIM):
    return [self.gym.find_actor_rigid_body_index(env, actor_handle, rb_name, domain)
            for env, actor_handle in zip(self.envs, self.actor_handles[actor_name])]

  def get_actor_rb_index(self, actor_name, rb_index, domain=gymapi.DOMAIN_SIM):
    return [self.gym.get_actor_rigid_body_index(env, actor_handle, rb_index, domain)
            for env, actor_handle in zip(self.envs, self.actor_handles[actor_name])]

  def get_obs_shape(self):
    return [[7],
            [self.num_dofs * 2]]

  def get_action_shape(self):
    return [self.num_dofs]


  def reset(self):
    if self.first_reset_called:
      self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.reset_dof_target_tensor))
      self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.reset_dof_state_tensor))
      self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.reset_root_body_tensor))
    else: self.first_reset_called = True

    self.terms = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self.truncs = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self.rewards = torch.zeros(self.num_envs, device=self.device)
    self._step()

    box_obs = self.rb_states[self.box_rb_idx, 0:7]
    return (box_obs, self.dof_states)


  def _step(self):
    # step the physics
    self.gym.simulate(self.sim)
    self.gym.fetch_results(self.sim, True)

    self.gym.refresh_dof_state_tensor(self.sim)
    self.gym.refresh_rigid_body_state_tensor(self.sim)
    self.gym.refresh_actor_root_state_tensor(self.sim)

    if not self.no_visual:
      self.gym.step_graphics(self.sim)


  def step(self, dof_targets):
    self.rewards = torch.zeros(self.num_envs, device=self.device)

    reset_envs = torch.nonzero(torch.logical_or(self.terms, self.truncs))
    if reset_envs.numel() != 0:
      print("resetting", reset_envs)

    dof_targets[reset_envs] = self.reset_dof_target_tensor[reset_envs]
    self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(dof_targets))

    if reset_envs.numel() != 0:
      dof_actor_resets = self.env_to_dof_actors[reset_envs]
      dar_int32 = dof_actor_resets.to(device=self.device, dtype=torch.int32)
      self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.reset_dof_state), gymtorch.unwrap_tensor(dar_int32), len(dar_int32))

      actor_resets = self.env_to_actors[reset_envs]
      ar_int32 = actor_resets.to(device=self.device, dtype=torch.int32)
      self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.reset_root_body), gymtorch.unwrap_tensor(ar_int32), len(ar_int32))

    self._step()
    self.calc_results(dof_targets)

    ret_terms = self.terms
    ret_truncs = self.truncs
    self.terms = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self.truncs = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    box_obs = self.rb_states[self.box_rb_idx, 0:7]

    return (box_obs, self.dof_states), self.rewards, ret_terms, ret_truncs


  def calc_results(self, dof_targets):
    lfinger_dist = torch.norm(self.rb_states[self.box_rb_idx, 0:3] \
                           -self.rb_states[self.lfing_rb_idx, 0:3], dim=1)
    rfinger_dist = torch.norm(self.rb_states[self.box_rb_idx, 0:3] \
                           -self.rb_states[self.rfing_rb_idx, 0:3], dim=1)
    box_height = self.target_height - self.rb_states[self.box_rb_idx, 2] 
    finished = box_height > self.target_height

    pose_norm = torch.norm(self.reset_dof_target_tensor - dof_targets, dim=1).mean()

    self.terms |= finished

    self.rewards = -(pose_norm + lfinger_dist + rfinger_dist + box_height)