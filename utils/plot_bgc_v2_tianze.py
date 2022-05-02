
from cProfile import label
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import *

from tensorboard.backend.event_processing import event_accumulator
from tensorboard_logger import configure, log_value
from torch import mean

def get_paths(path):
    paths = []
    for file_name in os.listdir(path):
        if file_name.startswith('PPO_'):
            paths.append(os.path.join(path, file_name))
    return paths

def read_tb_file(path):
    res_path = None
    for checkpoint_name in os.listdir(path):
        if checkpoint_name.startswith('events.out.tfevents'):
            res_path = os.path.join(path, checkpoint_name)

    if not res_path:
        return None

    ea = event_accumulator.EventAccumulator(res_path)
    ea.Reload()

    return ea


def handle_tb_file(ea, handle_key):
    res = {}
    #print(ea.scalars.Keys())
    for key in ea.scalars.Keys():
        #print(key)
        if key in handle_key:
            value = [scalar.value for scalar in ea.scalars.Items(key)]
            value = np.array(value)

            res[key] = value
    return res


def smooth(data, weight=0.5):
        smoothed = []
        last = data[0]
        for da in data:
            smooth_val = last * weight + (1 - weight) * da
            smoothed.append(smooth_val)
            last = smooth_val
        return np.array(smoothed)


def plot_4dr():
    metric = ['ray/tune/episode_reward_mean', 'ray/tune/info/num_steps_trained']

    # path
    origin_path = '../resource/new_data/m_env/origin_120_m_dynamic/'
    bgc_path = '../resource/new_data/m_env/bgc_120_m_dynamic/'

    # font 
    content_font = FontProperties(
        fname='/Users/zhoutianze/miniforge3/lib/python3.9/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf',
        size=15
    )

    title_font = FontProperties(
        fname='/Users/zhoutianze/miniforge3/lib/python3.9/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf',
        size=15
    )

    # tb
    origin_paths = get_paths(origin_path)
    bgc_paths = get_paths(bgc_path)
    
    origin_tbs = [read_tb_file(path) for path in origin_paths]
    bgc_tbs = [read_tb_file(path) for path in bgc_paths]

    # curve data
    origin_reses = []
    for tb in origin_tbs:
        origin_reses.append(handle_tb_file(tb, metric))
    
    bgc_reses = []
    for tb in bgc_tbs:
        bgc_reses.append(handle_tb_file(tb, metric))

    # calcualte min_t
    min_t = 104 #float('inf')
    metric_key = metric[-1]
    for origin_res in origin_reses:
        min_t = min(min_t, origin_res[metric_key].shape[0])
    for bgc_res in bgc_reses:
        min_t = min(min_t, bgc_res[metric_key].shape[0])
    t = bgc_reses[0][metric_key][:min_t]
    

    # plot
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    #plt.style.use('seaborn')
    
    metric_key = metric[0]
    temp_origin_data = []
    for origin_reses in origin_reses:
        temp_origin_data.append(smooth(origin_reses[metric_key][:min_t]))
    temp_origin_data = np.stack(temp_origin_data, axis=0)

    min_origin_data = temp_origin_data.min(axis=0)
    max_origin_data = temp_origin_data.max(axis=0)
    mean_origin_data = temp_origin_data.mean(axis=0)

    ax.plot(t, mean_origin_data, linewidth=2, label='原始算法', c='#BA3659')
    ax.fill_between(t, min_origin_data, max_origin_data, alpha=0.3, facecolor='#BA3659')

    temp_bgc_data = []
    for bgc_res in bgc_reses:
        temp_bgc_data.append(smooth(bgc_res[metric_key][:min_t]))
    temp_bgc_data = np.stack(temp_bgc_data, axis=0)

    min_bgc_data = temp_bgc_data.min(axis=0)
    max_bgc_data = temp_bgc_data.max(axis=0)
    mean_bgc_data = temp_bgc_data.mean(axis=0)

    ax.plot(t, mean_bgc_data, linewidth=2, label='邻域共识算法', c='#566069')
    ax.fill_between(t, min_bgc_data, max_bgc_data, alpha=0.3, facecolor='#566069')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # setting
    plt.grid(ls='--', alpha=0.5)
    plt.xlabel('训练步数(step)', FontProperties=content_font)
    plt.ylabel('多无人机平均奖励', FontProperties=content_font)
    plt.legend(prop=content_font, loc='lower right', fontsize=26)
    # plt.title('多无人机协同定位训练曲线（动态目标）',FontProperties=title_font)
    plt.savefig('./res/bgc_location_dynamic_target.pdf', dpi=500, format='pdf')
    plt.show()

def plot_4r():
    metric = ['ray/tune/episode_reward_mean', 'ray/tune/info/num_steps_trained']

    # path
    origin_path = '../resource/new_data/m_env/origin_120_m_fix/'
    bgc_path = '../resource/new_data/m_env/bgc_120_m_fix/'

    # font 
    content_font = FontProperties(
        fname='/Users/zhoutianze/miniforge3/lib/python3.9/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf',
        size=15
    )

    title_font = FontProperties(
        fname='/Users/zhoutianze/miniforge3/lib/python3.9/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf',
        size=15
    )

    # tb
    origin_paths = get_paths(origin_path)
    bgc_paths = get_paths(bgc_path)
    
    origin_tbs = [read_tb_file(path) for path in origin_paths]
    bgc_tbs = [read_tb_file(path) for path in bgc_paths]

    # curve data
    origin_reses = []
    for tb in origin_tbs:
        origin_reses.append(handle_tb_file(tb, metric))
    
    bgc_reses = []
    for tb in bgc_tbs:
        bgc_reses.append(handle_tb_file(tb, metric))

    # calcualte min_t
    min_t = 104 #float('inf')
    metric_key = metric[-1]
    for origin_res in origin_reses:
        min_t = min(min_t, origin_res[metric_key].shape[0])
    for bgc_res in bgc_reses:
        min_t = min(min_t, bgc_res[metric_key].shape[0])
    t = bgc_reses[0][metric_key][:min_t]
    

    # plot
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    #plt.style.use('seaborn')
    
    metric_key = metric[0]
    temp_origin_data = []
    for origin_reses in origin_reses:
        temp_origin_data.append(smooth(origin_reses[metric_key][:min_t]))
    temp_origin_data = np.stack(temp_origin_data, axis=0)

    min_origin_data = temp_origin_data.min(axis=0)
    max_origin_data = temp_origin_data.max(axis=0)
    mean_origin_data = temp_origin_data.mean(axis=0)

    ax.plot(t, mean_origin_data, linewidth=2, label='原始算法', c='#BA3659')
    ax.fill_between(t, min_origin_data, max_origin_data, alpha=0.3, facecolor='#BA3659')

    temp_bgc_data = []
    for bgc_res in bgc_reses:
        temp_bgc_data.append(smooth(bgc_res[metric_key][:min_t]))
    temp_bgc_data = np.stack(temp_bgc_data, axis=0)

    min_bgc_data = temp_bgc_data.min(axis=0)
    max_bgc_data = temp_bgc_data.max(axis=0)
    mean_bgc_data = temp_bgc_data.mean(axis=0)

    ax.plot(t, mean_bgc_data, linewidth=2, label='邻域共识算法', c='#566069')
    ax.fill_between(t, min_bgc_data, max_bgc_data, alpha=0.3, facecolor='#566069')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # setting
    plt.grid(ls='--', alpha=0.5)
    plt.xlabel('训练步数(step)', FontProperties=content_font)
    plt.ylabel('多无人机平均奖励', FontProperties=content_font)
    plt.legend(prop=content_font, loc='lower right', fontsize=26)
    # plt.title('多无人机协同定位训练曲线（动态目标）',FontProperties=title_font)
    plt.savefig('./res/bgc_location_fix_target.pdf', dpi=500, format='pdf')
    plt.show()

def plot_attack():
    metric = ['ray/tune/episode_reward_mean', 'ray/tune/info/num_steps_trained']

    # path
    origin_path = '../resource/new_data/c_env/origin_70_random/'
    bgc_path = '../resource/new_data/c_env/bgc_70_random/'

    # font 
    content_font = FontProperties(
        fname='/Users/zhoutianze/miniforge3/lib/python3.9/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf',
        size=15
    )

    title_font = FontProperties(
        fname='/Users/zhoutianze/miniforge3/lib/python3.9/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf',
        size=15
    )

    # tb
    origin_paths = get_paths(origin_path)
    bgc_paths = get_paths(bgc_path)
    
    origin_tbs = [read_tb_file(path) for path in origin_paths]
    bgc_tbs = [read_tb_file(path) for path in bgc_paths]

    # curve data
    origin_reses = []
    for tb in origin_tbs:
        origin_reses.append(handle_tb_file(tb, metric))
    
    bgc_reses = []
    for tb in bgc_tbs:
        bgc_reses.append(handle_tb_file(tb, metric))

    # calcualte min_t
    min_t = 272 #float('inf')
    metric_key = metric[-1]
    for origin_res in origin_reses:
        min_t = min(min_t, origin_res[metric_key].shape[0])
        print(min_t)
    for bgc_res in bgc_reses:
        min_t = min(min_t, bgc_res[metric_key].shape[0])
        print(min_t)
    t = bgc_reses[0][metric_key][:min_t]
    

    # plot
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    #plt.style.use('seaborn')
    
    metric_key = metric[0]
    temp_origin_data = []
    for origin_reses in origin_reses:
        temp_origin_data.append(smooth(origin_reses[metric_key][:min_t]))
    temp_origin_data = np.stack(temp_origin_data, axis=0)

    min_origin_data = temp_origin_data.min(axis=0)
    max_origin_data = temp_origin_data.max(axis=0)
    mean_origin_data = temp_origin_data.mean(axis=0)

    ax.plot(t, mean_origin_data, linewidth=2, label='原始算法', c='#BA3659')
    ax.fill_between(t, min_origin_data, max_origin_data, alpha=0.3, facecolor='#BA3659')

    temp_bgc_data = []
    for bgc_res in bgc_reses:
        temp_bgc_data.append(smooth(bgc_res[metric_key][:min_t]))
    temp_bgc_data = np.stack(temp_bgc_data, axis=0)

    min_bgc_data = temp_bgc_data.min(axis=0)
    max_bgc_data = temp_bgc_data.max(axis=0)
    mean_bgc_data = temp_bgc_data.mean(axis=0)

    ax.plot(t, mean_bgc_data, linewidth=2, label='邻域共识算法', c='#566069')
    ax.fill_between(t, min_bgc_data, max_bgc_data, alpha=0.3, facecolor='#566069')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # setting
    plt.grid(ls='--', alpha=0.5)
    plt.xlabel('训练步数(step)', FontProperties=content_font)
    plt.ylabel('多无人机平均奖励', FontProperties=content_font)
    plt.legend(prop=content_font, loc='upper left', fontsize=26)
    # plt.title('多无人机协同定位训练曲线（动态目标）',FontProperties=title_font)
    plt.savefig('./res/bgc_attack.pdf', dpi=500, format='pdf')
    plt.show()


def plot_attack_complete_vision():
    metric = ['ray/tune/episode_reward_mean', 'ray/tune/info/num_steps_trained']

    # path
    origin_path = '../resource/new_data/c_env/origin_1000_fix/'
    bgc_path = '../resource/new_data/c_env/bgc_1000_fix/'

    # font 
    content_font = FontProperties(
        fname='/Users/zhoutianze/miniforge3/lib/python3.9/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf',
        size=15
    )

    title_font = FontProperties(
        fname='/Users/zhoutianze/miniforge3/lib/python3.9/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf',
        size=15
    )

    # tb
    origin_paths = get_paths(origin_path)
    bgc_paths = get_paths(bgc_path)
    
    origin_tbs = [read_tb_file(path) for path in origin_paths]
    bgc_tbs = [read_tb_file(path) for path in bgc_paths]

    # curve data
    origin_reses = []
    for tb in origin_tbs:
        origin_reses.append(handle_tb_file(tb, metric))
    
    bgc_reses = []
    for tb in bgc_tbs:
        bgc_reses.append(handle_tb_file(tb, metric))

    # calcualte min_t
    min_t = 477 #float('inf')
    metric_key = metric[-1]
    for origin_res in origin_reses:
        min_t = min(min_t, origin_res[metric_key].shape[0])
    for bgc_res in bgc_reses:
        min_t = min(min_t, bgc_res[metric_key].shape[0])
    t = bgc_reses[0][metric_key][:min_t]
    

    # plot
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    #plt.style.use('seaborn')
    
    metric_key = metric[0]
    temp_origin_data = []
    for origin_reses in origin_reses:
        temp_origin_data.append(smooth(origin_reses[metric_key][:min_t]))
    temp_origin_data = np.stack(temp_origin_data, axis=0)

    min_origin_data = temp_origin_data.min(axis=0)
    max_origin_data = temp_origin_data.max(axis=0)
    mean_origin_data = temp_origin_data.mean(axis=0)

    ax.plot(t, mean_origin_data, linewidth=2, label='原始算法', c='#BA3659')
    ax.fill_between(t, min_origin_data, max_origin_data, alpha=0.3, facecolor='#BA3659')

    temp_bgc_data = []
    for bgc_res in bgc_reses:
        temp_bgc_data.append(smooth(bgc_res[metric_key][:min_t]))
    temp_bgc_data = np.stack(temp_bgc_data, axis=0)

    min_bgc_data = temp_bgc_data.min(axis=0)
    max_bgc_data = temp_bgc_data.max(axis=0)
    mean_bgc_data = temp_bgc_data.mean(axis=0)

    ax.plot(t, mean_bgc_data, linewidth=2, label='邻域共识算法', c='#566069')
    ax.fill_between(t, min_bgc_data, max_bgc_data, alpha=0.3, facecolor='#566069')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # setting
    plt.grid(ls='--', alpha=0.5)
    plt.xlabel('训练步数(step)', FontProperties=content_font)
    plt.ylabel('多无人机平均奖励', FontProperties=content_font)
    plt.legend(prop=content_font, loc='upper left', fontsize=26)
    # plt.title('多无人机协同定位训练曲线（动态目标）',FontProperties=title_font)
    plt.savefig('./res/bgc_attack_complete_vision.pdf', dpi=500, format='pdf')
    plt.show()

if __name__ == '__main__':
    # location
    #plot_4dr()
    #plot_4r()

    # attack
    plot_attack()
    plot_attack_complete_vision()