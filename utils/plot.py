import os
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=25)
matplotlib.rc('ytick', labelsize=25)

color = ['-ro', 'k-', 'b-', 'g-']


def plot_rewards(episodes, rewards, path):
    # File Name
    folder_name = os.path.join(path, 'plots')
    file_name = os.path.join(folder_name, 'rewards')

    font_size = 32

    # Figure
    plt.figure(figsize=(25, 10))

    # x-axis
    x_plt = range(0, episodes)

    # y-axis
    rewards_smoothed = smooth(rewards)
    y_plt = rewards_smoothed[range(0, episodes)]
    plt.plot(x_plt, y_plt.to("cpu"), color[0])

    plt.xlabel('Number of Episodes', fontsize=font_size)
    plt.ylabel('Reward', fontsize=font_size)
    plt.title("Rewards - per Episodes", fontsize=font_size)
    plt.savefig(file_name)


def plot_returns(episodes, returns, path):
    # File Name
    folder_name = os.path.join(path, 'plots')
    file_name = os.path.join(folder_name, 'returns')

    font_size = 32

    # Figure
    plt.figure(figsize=(25, 10))

    # x-axis
    x_plt = range(0, episodes)

    # y-axis
    returns_smoothed = smooth(returns)
    y_plt = returns_smoothed[range(0, episodes)]
    plt.plot(x_plt, y_plt.to("cpu"), color[0])

    plt.xlabel('Number of Episodes', fontsize=font_size)
    plt.ylabel('Return', fontsize=font_size)
    plt.title("Returns - per Episodes", fontsize=font_size)
    plt.savefig(file_name)


def plot_fps(episodes, fps, path):
    # File Name
    folder_name = os.path.join(path, 'plots')
    file_name = os.path.join(folder_name, 'fps')

    font_size = 32

    # Figure
    plt.figure(figsize=(25, 10))

    # x-axis
    x_plt = range(0, episodes)

    # y-axis
    fps_smoothed = smooth(fps)
    y_plt = fps_smoothed[range(0, episodes)]
    plt.plot(x_plt, y_plt.to("cpu"), color[0])

    plt.xlabel('Number of Episodes', fontsize=font_size)
    plt.ylabel('Frames per Second', fontsize=font_size)
    plt.title("Frames per Seconds - per Episodes", fontsize=font_size)
    plt.savefig(file_name)


def plot_episode_lenghts(episodes, episode_lengths, path):
    # File Name
    folder_name = os.path.join(path, 'plots')
    file_name = os.path.join(folder_name, 'episode_lengths')

    font_size = 32

    # Figure
    plt.figure(figsize=(25, 10))

    # x-axis
    x_plt = range(0, episodes)

    # y-axis
    episode_lengths_smoothed = smooth(episode_lengths)
    y_plt = episode_lengths_smoothed[range(0, episodes)]
    plt.plot(x_plt, y_plt.to("cpu"), color[0])

    plt.xlabel('Number of Episodes', fontsize=font_size)
    plt.ylabel('Episode Length', fontsize=font_size)
    plt.title("Episode Lengths - per Episodes", fontsize=font_size)
    plt.savefig(file_name)


#**********************
# Exponential smoothing
#**********************
def smooth(data, weight=0.99):
    last = data[0]
    smoothed = torch.zeros(len(data))
    idx = 0

    for datum in data:
        smoothed_val = last * weight + (1 - weight) * datum
        smoothed[idx] = smoothed_val
        last = smoothed_val
        idx = idx + 1

    return smoothed
