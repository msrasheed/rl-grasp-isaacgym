import torch
import numpy as np

import tqdm
import time

class TestRunner:
  def __init__(self, gym , sim, wviewer, env, actor, writer, run_params):
    self.gym = gym
    self.sim = sim
    self.wviewer = wviewer

    self.env = env
    self.actor = actor

    self.writer = writer

    self.run_params = run_params

  def run(self):
    self.actor.eval()
    with torch.no_grad():
      while not self.wviewer.check_viewer_closed():
        obs = self.env.reset()

        for _ in range(self.run_params["episode_secs"] * self.run_params["num_frames_sec"]):
          dof_targets, target_logprobs, values = self.actor.get_action_value(*obs)  
          next_obs, rewds, terms, truncs = self.env.step(dof_targets)

          if not self.wviewer.no_visual:
            self.wviewer.step()

          obs = next_obs
          if self.wviewer.check_viewer_closed():
            break

class TrainRunner:
  def __init__(self, gym , sim, wviewer, env, actor, writer,
               device, train_params, optimizer, save_actor):
    self.gym = gym
    self.sim = sim
    self.wviewer = wviewer

    self.env = env
    self.actor = actor

    self.writer = writer

    self.device = device
    self.params = train_params
    self.optimizer = optimizer
    self.save_actor = save_actor


  def run(self):
    num_envs = self.env.num_envs
    steps_train = self.params["steps_train"]
    epochs = self.params["epochs"]
    minibatch_size = self.params["minibatch_size"]

    gae_gamma = self.params["gae_gamma"]
    gae_lambda = self.params["gae_lambda"]

    clip_coef = self.params["clip_coef"]
    ent_coef = self.params["ent_coef"]
    vf_coef = self.params["vf_coef"]
    max_grad_norm = self.params["max_grad_norm"]

    obs_shapes = self.env.get_obs_shape()
    act_shape = self.env.get_action_shape()

    obss = [torch.zeros(steps_train, num_envs, *shape).to(device=self.device) for shape in obs_shapes]
    acts = torch.zeros(steps_train, num_envs, *act_shape).to(device=self.device)
    acts_probs = torch.zeros(steps_train, num_envs).to(device=self.device)
    vals = torch.zeros(steps_train+1, num_envs).to(device=self.device)
    rewards = torch.zeros(steps_train, num_envs).to(device=self.device)
    dones = torch.zeros(steps_train, num_envs).to(device=self.device)


    global_step = 0
    start_time = time.time()
    max_return = float('-inf')

    for episode in tqdm.tqdm(range(self.params["num_episodes"])):
    
      obs = self.env.reset()
      num_terms = 0
      tot_returns = 0

      num_steps = (self.params["episode_secs"] * self.params["num_frames_sec"]) \
                  // self.params["steps_train"]

      for train_step in range(num_steps):
      
        self.actor.eval()
        with torch.no_grad():
          i = 0
          while i < steps_train and not self.wviewer.check_viewer_closed():
            frame_no = self.gym.get_frame_count(self.sim)
            global_step += 1 * num_envs

            # compute action
            dof_targets, target_logprobs, values = self.actor.get_action_value(*obs)  

            # step environment
            next_obs, rewds, terms, truncs = self.env.step(dof_targets)
            # print(rewds)

            for ob, hist in zip(obs, obss):
              hist[i] = ob 
            acts[i] = dof_targets
            acts_probs[i] = target_logprobs.flatten()
            vals[i] = values.flatten()
            rewards[i] = rewds
            dones[i] = torch.logical_or(terms, truncs)

            num_terms += torch.count_nonzero(terms)

            if not self.wviewer.no_visual:
              self.wviewer.step()

            i += 1
            obs = next_obs

          if self.wviewer.check_viewer_closed():
            break
          
          # next values for bootstrapping
          _, _, values = self.actor.get_action_value(*obs)
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
        b_obss = [o.reshape(-1, *o.shape[2:]) for o in obss]
        b_acts = acts.reshape(-1, *acts.shape[2:])
        b_acts_probs = acts_probs.reshape(-1, *acts_probs.shape[2:])
        b_vals = values.flatten()
        b_returns = returns.flatten()
        b_dones = dones.flatten()
        b_advantages = advantages.flatten()

        self.actor.train()

        b_inds = np.arange(b_dones.numel())
        clipfracs = []
        for epoch in range(epochs):
          np.random.shuffle(b_inds) # in-place
          for start in range(0, b_dones.numel(), minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            # newactprob, entropy, newvalue = self.actor(b_img_obs[mb_inds], b_dof_obs[mb_inds], b_acts[mb_inds])
            newactprob, entropy, newvalue = self.actor(*[o[mb_inds] for o in b_obss], b_acts[mb_inds])
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

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
            self.optimizer.step()

        y_pred = b_vals.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true-y_pred) / var_y

        self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
        self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        self.writer.add_scalar("diagnostic/value_mean", b_vals.mean().item(), global_step)
        self.writer.add_scalar("diagnostic/return_mean", b_returns.mean().item(), global_step)
        self.writer.add_scalar("diagnostic/mean_diff", torch.square((b_returns-b_vals)).mean().item(), global_step)


      self.writer.add_scalar("progress/num_terms", num_terms.item(), global_step)
      self.writer.add_scalar("progress/tot_returns", tot_returns.item(), global_step)

      if tot_returns.item() > max_return:
        self.save_actor("max")

      if episode % 100 == 0:
        self.save_actor(str(episode))

      if self.wviewer.check_viewer_closed():
        break
      
    self.save_actor("end")