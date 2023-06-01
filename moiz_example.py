
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym import torch_utils

import torch
import numpy as np

import math

import grasp_policy


# get access to gymapi interface
gym = gymapi.acquire_gym()

# default arguments
custom_parameters = [{"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"}]
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
cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)

# subscribe to input events
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "cam_cap")

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
num_per_row = int(math.sqrt(args.num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

envs = list()
frankas = list()
cameras = list()
cam_tensors_arr = list()
franka_actor_idx = list()
# rigid body (rb) indexes
hand_rb_index = list()
box_rb_index = list()

print("Creating %d environments" % args.num_envs)
for i in range(args.num_envs):
  env = gym.create_env(sim, env_lower, env_upper, num_per_row)
  envs.append(env)

  # add franka
  franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 1)
  frankas.append(franka_handle)
  hand_rb_index.append(gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM))
  franka_actor_idx.append(gym.get_actor_index(env, franka_handle, gymapi.DOMAIN_SIM))

  # set dof props (the drives)
  gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)
  # set dof states (sets position, but doesn't stay if targets don't change)
  gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)
  # set dof targets (otherwise robot just stands up to default targets)
  gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

  # add table
  table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

  # add box - random position/rotation on table
  box_pose.p.x = table_pose.p.x # + np.random.uniform(-0.2, 0.1)
  box_pose.p.y = table_pose.p.y # + np.random.uniform(-0.3, 0.3)
  box_pose.p.z = table_dims.z + 0.5 * box_size
  box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
  box_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 0)
  color = gymapi.Vec3(*np.random.uniform(0, 1, 3))
  gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
  box_rb_index.append(gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM))

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
policyNet.eval()

# collect dof tensors (frankas dof, other objects don't have dofs)
franka_actor_idx = torch.tensor(franka_actor_idx)
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
print("dof_states shape", dof_states.shape)
dof_state_inpol = dof_states.view(args.num_envs, franka_num_dofs*2)
dof_state_reset = dof_states.view(args.num_envs, franka_num_dofs, 2)

# collect rb tensor for reward calculations later
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)
print("rb_states shape", rb_states.shape)

# sim params
steps_train = 50
episode_secs = 20

# storage tensors
# img_obs = torch.zeros(args.num_envs, )
# dof_obs
# acts
# acts_probs
# vals
# rewards
# terms

print("dof indexes", [gym.get_actor_dof_index(envs[0], frankas[0], i, gymapi.DOMAIN_SIM) for i in range(9)])
print("num actors", gym.get_sim_actor_count(sim))

frame_wait = 50
while not gym.query_viewer_has_closed(viewer):
  frame_no = gym.get_frame_count(sim)
  # step the physics
  gym.simulate(sim)
  gym.fetch_results(sim, True)

  gym.refresh_dof_state_tensor(sim)
  gym.refresh_rigid_body_state_tensor(sim)

  

  cam_cap = False
  for evt in gym.query_viewer_action_events(viewer):
    if evt.action == "cam_cap" and evt.value > 0:
      cam_cap = True

  # update viewer
  gym.step_graphics(sim)
  # render sensors and refresh camera tensors
  gym.render_all_camera_sensors(sim)
  gym.start_access_image_tensors(sim)

  cam_tensors = torch.stack(cam_tensors_arr)
  cam_fails = torch.unique(torch.nonzero(torch.isinf(cam_tensors))[:, 0])
  cam_tensors[cam_fails] = torch.zeros(len(cam_fails), 1, 128, 128).to(device=device)

  with torch.no_grad():
    dof_targets, _, _ = policyNet.get_action_value(cam_tensors, dof_state_inpol)  

  gym.end_access_image_tensors(sim)

  if cam_cap: cam_fails = torch.tensor([4,7,24])

  dof_targets[cam_fails] = default_dof_pos_tensor
  gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(dof_targets))

  if cam_fails.numel() != 0:
    print("resetting", cam_fails, "on frame", frame_no)
    new_dof_state = torch.zeros_like(dof_state_reset)
    new_dof_state[cam_fails, :, 0] = default_dof_pos_tensor
    new_dof_state = new_dof_state.view(args.num_envs * franka_num_dofs, 2)
    franka_fails = franka_actor_idx[cam_fails]
    franka_fails_int32 = franka_fails.to(device=device, dtype=torch.int32)
    gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(new_dof_state), gymtorch.unwrap_tensor(franka_fails_int32), len(franka_fails_int32))


  # draw viewer
  gym.draw_viewer(viewer, sim, True)
  # wait for dt to elapse in real time
  gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)