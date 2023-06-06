
class IsaacGymViewerWrapper:
  def __init__(self, gym, sim, viewer, no_visual=False):
    self.gym = gym
    self.sim = sim
    self.viewer = viewer
    self.no_visual = no_visual

    if no_visual: 
      self.check_viewer_closed = lambda: False
    else:
      self.check_viewer_closed = self._viewer_close_check
  
  def step(self):
    if not self.no_visual:
      self.gym.draw_viewer(self.viewer, self.sim, True)
      self.gym.sync_frame_time(self.sim)


  def _viewer_close_check(self):
    return self.gym.query_viewer_has_closed(self.viewer)