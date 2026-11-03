# NEW — highway adaptation layer, not in upstream (WocaR-RL/WocaR-DQN/SA-DQN).
# Bridges highway-env (gymnasium, Kinematics obs, 5-tuple API) to SA-DQN's 4-value Atari API.
import numpy as np


try:
    import gymnasium as gym
except ImportError:
    import gym


class LegacyHighwayEnv:
    """Expose highway-env through the four-value API used by SA-DQN."""

    def __init__(self, env_id, env_config=None, render_mode=None):
        import highway_env  # noqa: F401

        make_kwargs = {'config': env_config or {}}
        if render_mode is not None:
            make_kwargs['render_mode'] = render_mode
        self.env = gym.make(env_id, **make_kwargs)
        self.action_space = self.env.action_space
        self._observation_space = self.env.observation_space
        self.observation_space = self._flatten_space(self._observation_space)

        if not hasattr(self.action_space, "n"):
            raise TypeError("The vanilla DQN baseline requires a discrete action space")

    @staticmethod
    def _flatten_space(space):
        if not hasattr(space, "shape"):
            raise TypeError("The highway observation space must be a Box-like space")

        low = np.asarray(space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(space.high, dtype=np.float32).reshape(-1)
        return type(space)(low=low, high=high, dtype=np.float32)

    @staticmethod
    def _flatten_observation(observation):
        return np.asarray(observation, dtype=np.float32).reshape(-1)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        observation = result[0] if isinstance(result, tuple) else result
        return self._flatten_observation(observation)

    def step(self, action):
        result = self.env.step(int(action))
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            observation, reward, done, info = result
        info = dict(info)
        info["crashed"] = bool(getattr(self.env.unwrapped.vehicle, "crashed", False))
        return self._flatten_observation(observation), reward, done, info

    def seed(self, seed=None):
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        self.env.reset(seed=seed)
        return [seed]

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()