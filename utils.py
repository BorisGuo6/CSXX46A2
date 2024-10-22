import gym
from gym import spaces


class DictToListWrapper(gym.Wrapper):
    def __init__(self, env):
        super(DictToListWrapper, self).__init__(env)
        if isinstance(env.observation_space, spaces.Dict):
            self.env_features = list(env.observation_space.keys())
        else:
            raise TypeError("Observation space must be of type gym.spaces.Dict.")

        self.observation_space = spaces.Box(
            low=-float('inf'), high=float('inf'), shape=(len(self.env_features),), dtype=float
        )

    def convert_state_to_list(self, state):
        return [state[feature] for feature in self.env_features]

    def reset(self):
        state = self.env.reset()
        state_desc = self.env.disc2state(state)
        return self.convert_state_to_list(state_desc), {}

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state_desc = self.env.disc2state(state)
        return self.convert_state_to_list(state_desc), reward, done, False, info


# ---


import matplotlib.pyplot as plt
from IPython.display import clear_output


def exponential_smoothing(data, alpha=0.1):
    """Compute exponential smoothing."""
    smoothed = [data[0]]  # Initialize with the first data point
    for i in range(1, len(data)):
        st = alpha * data[i] + (1 - alpha) * smoothed[-1]
        smoothed.append(st)
    return smoothed


def live_plot(data_dict):
    """Plot the live graph with multiple subplots."""

    plt.style.use('ggplot')
    n_plots = len(data_dict)
    fig, axes = plt.subplots(nrows=n_plots, figsize=(7, 4 * n_plots), squeeze=False)
    plt.subplots_adjust(hspace=0.5)
    plt.ion()
    clear_output(wait=True)

    for ax, (label, data) in zip(axes.flatten(), data_dict.items()):
        ax.clear()
        ax.plot(data, label=label, color="yellow", linestyle='--')
        # Compute and plot moving average for total reward
        if len(data) > 0:
            ma = exponential_smoothing(data)
            ma_idx_start = len(data) - len(ma)
            ax.plot(range(ma_idx_start, len(data)), ma, label="Smoothed Value",
                    linestyle="-", color="purple", linewidth=2)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(loc='upper left')

    plt.show()
