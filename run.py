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
import hand_cam_box_env as hcbe
import test_train_runner as ttr
import viewer_wrapper as vw



# get access to gymapi interface
gym = gymapi.acquire_gym()

# default arguments
custom_parameters = [
  {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
  {"name": "--test", "action": "store_true", "help": "To run in test mode, no training"},
  {"name": "--load", "type": str, "help": "Weights to load"},
  {"name": "--no_visual", "action": "store_true", "help": "Whether to render display or not"},
  {"name": "--no_writer", "action": "store_true", "help": "Don't log anything to tensorboard"}
  ]

args = gymutil.parse_arguments(custom_parameters=custom_parameters)
# default: Namespace(compute_device_id=0, flex=False, graphics_device_id=0, num_threads=0, 
#                    physics_engine=SimType.SIM_PHYSX, physx=False, pipeline='gpu', sim_device='cuda:0', 
#                    sim_device_type='cuda', slices=0, subscenes=0, use_gpu=True, use_gpu_pipeline=True)

# set torch device
device = args.sim_device if args.use_gpu_pipeline else "cpu"
print("device", device)

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
num_envs = args.num_envs

# create viewer
if not args.no_visual:
  cam_props = gymapi.CameraProperties()
  viewer = gym.create_viewer(sim, cam_props)
else:
  viewer = None
wviewer = vw.IsaacGymViewerWrapper(gym, sim, viewer, args.no_visual)

  
raw_env = hcbe.HandCamBoxEnv(gym, sim, 
                               num_envs=args.num_envs,
                               device=device)
raw_env.create()

run_params = dict(
  episode_secs = 20,
  num_frames_sec = 60
)
train_params = dict(
  gae_gamma = 0.99,
  gae_lambda = 0.95,
  epochs = 3,
  minibatch_size = 1000,
  clip_coef = 0.2,
  ent_coef = 0,
  vf_coef = 0.5,
  max_grad_norm = .5,

  num_episodes = 1000,
  steps_train = 50,
  episode_secs = 20,
  num_frames_sec = 60
)

assert ((train_params["episode_secs"] * train_params["num_frames_sec"]) \
        % train_params["steps_train"]) == 0, \
        "episode num frames not divisible by num train frames"


# create policy network
policyNet = grasp_policy.Policy().to(device=device)
optimizer = optim.Adam(policyNet.parameters(), lr=3e-4)


env = igenv.NormalizeWrapper(raw_env,
                             obs_shapes=raw_env.get_obs_shape(),
                             gamma=train_params["gae_gamma"])

# tensorboard logger
if args.no_writer:
  writer = None
else:
  writer = SummaryWriter()

def save_actor(name_prefix):
  path_name = os.path.join(writer.log_dir, name_prefix)
  torch.save(policyNet.state_dict(), path_name + ".pth")
  env.save_obs_tensors(path_name)

def load_actor(name_prefix):
  policyNet.load_state_dict(torch.load(name_prefix + ".pth"))
  env.load_obs_tensors(name_prefix)
  
if args.no_visual:
  def sigint_handler(sig, frame):
    save_actor("end")
    gym.destroy_sim(sim)
    exit()
  signal.signal(signal.SIGINT, sigint_handler)

if args.load:
  load_actor(args.load)

if args.test:
  runner = ttr.TestRunner(
    gym, sim, wviewer, env, policyNet, writer, run_params
  )
else:
  runner = ttr.TrainRunner(
    gym, sim, wviewer, env, policyNet, writer,
    device, train_params, optimizer, save_actor
  )

runner.run()


if not args.no_visual:
  gym.destroy_viewer(viewer)
gym.destroy_sim(sim)