
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym import torch_utils

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tqdm

import os
import math
import time
import signal

import grasp_policy
import igenv
import running_mean_std as rms
import util


# get access to gymapi interface
gym = gymapi.acquire_gym()

# default arguments
custom_parameters = [
  {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
  {"name": "--no_visual", "action": "store_true", "help": "Whether to render display or not"}
  ]

args = gymutil.parse_arguments(custom_parameters=custom_parameters)
# default: Namespace(compute_device_id=0, flex=False, graphics_device_id=0, num_threads=0, 
#                    physics_engine=SimType.SIM_PHYSX, physx=False, pipeline='gpu', sim_device='cuda:0', 
#                    sim_device_type='cuda', slices=0, subscenes=0, use_gpu=True, use_gpu_pipeline=True)

# set torch device
device = args.sim_device if args.use_gpu_pipeline else "cpu"

# sets sim params
def get_sim_params(args):
  # get default set of parameters
  sim_params = gymapi.SimParams()

  # set common parameters
  sim_params.dt = 1 / 60
  sim_params.substeps = 2
  sim_params.up_axis = gymapi.UP_AXIS_Z
  sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
  sim_params.use_gpu_pipeline = args.use_gpu_pipeline
  # sim_params.num_client_threads
  # sim_params.stress_visualization
  # sim_params.stress_visualization_max
  # sim_params.stress_visualisation_min

  # physX robust rigid body and articulation simulation
  # that runs on CPU or GPU, support tensor API
  # set PhysX-specific parameters
  sim_params.physx.use_gpu = args.use_gpu
  sim_params.physx.solver_type = 1
  sim_params.physx.num_position_iterations = 8
  sim_params.physx.num_velocity_iterations = 1
  sim_params.physx.contact_offset = 0.001
  sim_params.physx.rest_offset = 0.0

  sim_params.physx.friction_offset_threshold = 0.001
  sim_params.physx.friction_correlation_distance = 0.0005
  sim_params.physx.num_threads = args.num_threads
  # sim_params.phsyx.always_use_articulations
  # sim_params.phsyx.bounce_threshold_velocity
  # sim_params.phsyx.contact_collection
  # sim_params.phsyx.default_buffer_size_multiplier
  # sim_params.phsyx.max_depenetration_velocity
  # sim_params.physx.max_gpu_contact_pairs
  # sim_params.physx.num_position_iterations
  # sim_params.phsyx.num_subscenes

  # Flex soft and rigid body simulation, runs on GPU, 
  # not fully suport Tensor API yet
  # set Flex-specific parameters
  sim_params.flex.solver_type = 5
  sim_params.flex.num_outer_iterations = 4
  sim_params.flex.num_inner_iterations = 20
  sim_params.flex.relaxation = 0.8
  sim_params.flex.warm_start = 0.5

  return sim_params

# create simulation
# "The sim object contains physics and graphics contexts that will allow you to load assets, create environments, and interact with the simulation."
sim_params = get_sim_params(args)
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# create viewer
if not args.no_visual:
  cam_props = gymapi.CameraProperties()
  viewer = gym.create_viewer(sim, cam_props)

  # subscribe to input events
  gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "cam_cap")
  
  def check_viewer_closed():
    return gym.query_viewer_has_closed(viewer)
  
else:
  def check_viewer_closed():
    return False


# Creating Ground Plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up, if y-up (0,1,0)
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

gym.add_ground(sim, plane_params)

# load franka asset
asset_root = "../isaacgym/assets"
asset_file = "urdf/franka_description/robots/franka_panda.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
# asset_options.disable_gravity = True
asset_options.flip_visual_attachments = True
franka_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# franka pose, for creating actor
franka_pose = gymapi.Transform()
franka_pose.p = gymapi.Vec3(0,0,0)

# get franka dof props and joint ranges
franka_dof_props = gym.get_asset_dof_properties(franka_asset)
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
franka_num_dofs = gym.get_asset_dof_count(franka_asset)
# print("num dofs", franka_num_dofs)
default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
default_dof_pos[:7] = franka_mids[:7]
# grippers open
default_dof_pos[7:] = franka_upper_limits[7:]

default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

default_dof_pos_tensor = torch_utils.to_torch(default_dof_pos, device=device)


# create table asset
table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# table pose
table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)


# create object asset
box_size = 0.045
box_asset = gym.create_box(sim, box_size, box_size, box_size)
box_pose = gymapi.Transform()

# configure env grid
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

envs = list()
frankas = list()
cameras = list()
cam_tensors_arr = list()
franka_actor_idx = list()
box_actor_idx = list()
# rigid body (rb) indexes
hand_rb_index = list()
box_rb_index = list()
# root rigid body collect
root_body_init = list()

print("Creating %d environments" % num_envs)
for i in range(num_envs):
  env = gym.create_env(sim, env_lower, env_upper, num_per_row)
  envs.append(env)

  # add franka
  franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 1)
  frankas.append(franka_handle)
  hand_rb_index.append(gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM))
  franka_actor_idx.append(gym.get_actor_index(env, franka_handle, gymapi.DOMAIN_SIM))
  root_body_init.append(util.transform_to_rb_tensor(franka_pose))

  # set dof props (the drives)
  gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)
  # set dof states (sets position, but doesn't stay if targets don't change)
  gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)
  # set dof targets (otherwise robot just stands up to default targets)
  gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

  # add table
  table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)
  root_body_init.append(util.transform_to_rb_tensor(table_pose))

  # add box - random position/rotation on table
  box_pose.p.x = table_pose.p.x # + np.random.uniform(-0.2, 0.1)
  box_pose.p.y = table_pose.p.y # + np.random.uniform(-0.3, 0.3)
  box_pose.p.z = table_dims.z + 0.5 * box_size
  box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
  box_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 0)
  color = gymapi.Vec3(*np.random.uniform(0, 1, 3))
  gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
  box_rb_index.append(gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM))
  box_actor_idx.append(gym.get_actor_index(env, box_handle, gymapi.DOMAIN_SIM))
  root_body_init.append(util.transform_to_rb_tensor(box_pose))

  # add camera
  camera_props = gymapi.CameraProperties()
  camera_props.width = 128
  camera_props.height = 128
  camera_props.enable_tensors = True
  camera_handle = gym.create_camera_sensor(env, camera_props)
  body_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
  local_transform = gymapi.Transform()
  local_transform.p = gymapi.Vec3(.04, 0, 0)
  local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(-90))
  gym.attach_camera_to_body(camera_handle, env, body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
  cameras.append(camera_handle)

  # get camera tensor
  cam_tensor = gym.get_camera_image_gpu_tensor(sim, env, camera_handle, gymapi.IMAGE_DEPTH)
  torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
  torch_cam_tensor = torch_cam_tensor.unsqueeze(0)
  cam_tensors_arr.append(torch_cam_tensor)

# prepare internal data structures for tensor API
# otherwise get gym cuda error: an illegal memory access was encountered
gym.prepare_sim(sim)

# create policy network
policyNet = grasp_policy.Policy().to(device=device)
optimizer = optim.Adam(policyNet.parameters(), lr=3e-4)


# collect dof tensors (frankas dof, other objects don't have dofs)
franka_actor_idx = torch.tensor(franka_actor_idx)
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
print("dof_states shape", dof_states.shape)
dof_state_inpol = dof_states.view(num_envs, franka_num_dofs*2)
dof_state_reset = dof_states.view(num_envs, franka_num_dofs, 2)

# collect rb tensor for reward calculations later
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)
print("rb_states shape", rb_states.shape)

# collect root state tensor for resets
_root_tensor = gym.acquire_actor_root_state_tensor(sim)
root_tensor = gymtorch.wrap_tensor(_root_tensor)


# setup reset vals
reset_dof_state_tensor = torch.zeros_like(dof_state_reset)
reset_dof_state_tensor[:, :, 0] = default_dof_pos_tensor
reset_dof_target_tensor = reset_dof_state_tensor[:, :, 0].clone()
reset_root_body_tensor = torch.stack(root_body_init).to(device)

# sim params
num_episodes = 1000
steps_train = 50
episode_secs = 20
num_frames_sec = 60
assert ((episode_secs * num_frames_sec) % steps_train) == 0, "episode num frames not divisible by num train frames"

target_height = table_dims.z + .2

gae_gamma = 0.99
gae_lambda = 0.95
epochs = 3
minibatch_size = 1000
clip_coef = 0.2
ent_coef = 0
vf_coef = 0.5
max_grad_norm = .5

raw_env = igenv.IsaacGymEnv(gym, sim, 
                        franka_num_dofs=franka_num_dofs,
                        target_height=target_height,
                        device=device, 
                        env_to_box=torch.tensor(box_actor_idx),
                        env_to_franka=franka_actor_idx,
                        box_rb_idx=box_rb_index,
                        hand_rb_idx=hand_rb_index,
                        dof_state_tensor=dof_state_inpol,
                        cam_tensors_arr=cam_tensors_arr,
                        rb_states=rb_states)

# storage tensors
img_obs = torch.zeros(steps_train, num_envs, 1, 128, 128).to(device=device)
dof_obs = torch.zeros(steps_train, num_envs, franka_num_dofs*2).to(device=device)
acts = torch.zeros(steps_train, num_envs, franka_num_dofs).to(device=device)
acts_probs = torch.zeros(steps_train, num_envs).to(device=device)
vals = torch.zeros(steps_train+1, num_envs).to(device=device)
rewards = torch.zeros(steps_train, num_envs).to(device=device)
dones = torch.zeros(steps_train, num_envs).to(device=device)

env = igenv.NormalizeWrapper(raw_env,
                             obs_shapes=[img_obs.shape[2:],
                                         dof_obs.shape[2:]],
                             gamma=gae_gamma)

# tensorboard logger
writer = SummaryWriter()
def save_actor(name_prefix):
  path_name = os.path.join(writer.log_dir, name_prefix)
  torch.save(policyNet.state_dict(), path_name + ".pth")
  env.save_obs_tensors(path_name)
  
if args.no_visual:
  def sigint_handler(sig, frame):
    save_actor("end")
    gym.destroy_sim(sim)
    exit()
  signal.signal(signal.SIGINT, sigint_handler)


global_step = 0
start_time = time.time()
max_return = float('-inf')

for episode in tqdm.tqdm(range(num_episodes)):

  obs = env.reset(reset_dof_target_tensor, reset_dof_state_tensor, reset_root_body_tensor)
  num_terms = 0
  tot_returns = 0

  for train_step in range((episode_secs * num_frames_sec) // steps_train):

    policyNet.eval()
    with torch.no_grad():
      i = 0
      while i < steps_train and not check_viewer_closed():
        frame_no = gym.get_frame_count(sim)
        global_step += 1 * num_envs

        cam_tensors, dof_states = obs

        # compute action
        dof_targets, target_logprobs, values = policyNet.get_action_value(cam_tensors, dof_states)  

        # step environment
        next_obs, rewds, terms, truncs = env.step(dof_targets)
        # print(rewds)

        img_obs[i] = cam_tensors
        dof_obs[i] = dof_states
        acts[i] = dof_targets
        acts_probs[i] = target_logprobs.flatten()
        vals[i] = values.flatten()
        rewards[i] = rewds
        dones[i] = torch.logical_or(terms, truncs)

        num_terms += torch.count_nonzero(terms)

        if not args.no_visual:
          # draw viewer
          gym.draw_viewer(viewer, sim, True)
          # wait for dt to elapse in real time
          gym.sync_frame_time(sim)

        i += 1
        obs = next_obs

      if check_viewer_closed():
        break

      # next values for bootstrapping
      cam_tensors, dof_states = obs
      _, _, values = policyNet.get_action_value(cam_tensors, dof_states)
      vals[i] = values.flatten()

      # calculate advantages
      not_last_step = 1 - dones
      advantages = torch.zeros_like(vals)
      for t in reversed(range(steps_train)):
        delta = rewards[t] + gae_gamma*not_last_step[t]*vals[t+1] - vals[t]
        advantages[t] = delta + gae_gamma*gae_lambda*not_last_step[t]*advantages[t+1]
    
      advantages = advantages[:steps_train]
      values = vals[:steps_train]
      returns = advantages + values

      tot_returns = returns.mean() + gae_gamma * tot_returns

    # out of torch.no_grad now
    # flatten the batch
    b_img_obs = img_obs.reshape(-1, *img_obs.shape[2:])
    b_dof_obs = dof_obs.reshape(-1, *dof_obs.shape[2:])
    b_acts = acts.reshape(-1, *acts.shape[2:])
    b_acts_probs = acts_probs.reshape(-1, *acts_probs.shape[2:])
    b_vals = values.flatten()
    b_returns = returns.flatten()
    b_dones = dones.flatten()
    b_advantages = advantages.flatten()

    policyNet.train()

    b_inds = np.arange(b_dones.numel())
    clipfracs = []
    for epoch in range(epochs):
      np.random.shuffle(b_inds) # in-place
      for start in range(0, b_dones.numel(), minibatch_size):
        end = start + minibatch_size
        mb_inds = b_inds[start:end]

        newactprob, entropy, newvalue = policyNet(b_img_obs[mb_inds], b_dof_obs[mb_inds], b_acts[mb_inds])
        logratio = newactprob - b_acts_probs[mb_inds]
        ratio = logratio.exp()

        with torch.no_grad():
          old_approx_kl = (-logratio).mean()
          approx_kl = ((ratio-1) - logratio).mean()
          clipfracs += [((ratio-1.0).abs() > clip_coef).float().mean().item()]

        mb_advantages = (-b_advantages[mb_inds])
        pg_loss1 = mb_advantages * ratio
        pg_loss2 = mb_advantages * torch.clamp(ratio, 1-clip_coef, 1+clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # no clipping
        v_loss = 0.5 * torch.pow(newvalue - b_returns[mb_inds], 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - ent_coef*entropy_loss + vf_coef*v_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policyNet.parameters(), max_grad_norm)
        optimizer.step()

    y_pred = b_vals.cpu().numpy()
    y_true = b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true-y_pred) / var_y
    
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    # print("SPS:", int(global_step / (time.time() - start_time)))
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    writer.add_scalar("diagnostic/value_mean", b_vals.mean().item(), global_step)
    writer.add_scalar("diagnostic/return_mean", b_returns.mean().item(), global_step)

  writer.add_scalar("progress/num_terms", num_terms.item(), global_step)
  writer.add_scalar("progress/tot_returns", tot_returns.item(), global_step)

  if tot_returns.item() > max_return:
    save_actor("max")

  if check_viewer_closed():
    break

save_actor("end")

if not args.no_visual:
  gym.destroy_viewer(viewer)
gym.destroy_sim(sim)