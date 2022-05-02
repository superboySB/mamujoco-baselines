
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy import interpolate


class MultiPlot(object):
    def __init__(self, path_data_csv, weight=0.8):
        self.path_data_csv = path_data_csv
        self.weight = weight

        self.len_path_data_csv = len(path_data_csv)
        self.w = int(self.len_path_data_csv ** (0.5) + 1)
        self.h = self.len_path_data_csv // self.w if self.len_path_data_csv % self.w == 0 else self.len_path_data_csv // self.w + 1
        if self.len_path_data_csv == 3:
            self.w = 3
            self.h = 1

        self.colors = ['c', 'g', 'b', 'm', 'r', 'y']
        self.algos = {}
        self.lines = {}

        self.data = {}
        self.get_value_data()
        self.data_keys = [k for k, v in self.data.items()]

    def get_value_data(self, data_slice=100, weight=100, bias=0):
        # 读取每一个csv场景
        for paths_scenario in self.path_data_csv:
            scenario_name = paths_scenario[0].split('/')[-2]
            scenario_value = {}
            min_slice = np.inf
            # 读取场景中每个算法的效果 
            for path in paths_scenario:
                name_algo = path.split('/')[-3]
                fullpaths = self.get_csv_path(path)
                if not fullpaths:
                    continue
                if name_algo not in self.algos:
                    self.algos[name_algo] = len(self.algos)

                value_data = [self.read_csv(fullpath) for fullpath in fullpaths]

                value_data_len = min([len(data) for data in value_data])
                value_data_slice = value_data_len // data_slice
                min_slice = value_data_slice if value_data_slice < min_slice else min_slice

                value_data = [self.smooth(data)[:value_data_len] for data in value_data]
                value_data = np.array(value_data) #* weight + bias
                scenario_value[name_algo] = value_data
            
            for scenario_value_key in scenario_value:
                scenario_value[scenario_value_key] = scenario_value[scenario_value_key][:, :min_slice*data_slice]
            self.data[scenario_name] = scenario_value

    def plot(self):
        if self.h == 1:
            self.figure, self.ax = plt.subplots(1, self.w, figsize=(24, 5))
        else:
            self.figure, self.ax = plt.subplots(self.h, self.w, figsize=(24, 12))
        
        for h in range(self.h):
            for w in range(self.w):
                if h * self.w + w > self.len_path_data_csv-1:
                    break
                self._plot(h, w)
        
        lines = [v for k, v in self.lines.items()]
        label = [k for k, v in self.lines.items()]

        self.figure.legend(lines, 
                           label,
                           loc="upper center",
                           ncol=len(lines),
                           prop={'size':13},
                           fontsize=16)
        plt.subplots_adjust(left=0.043, 
                            bottom=0.083, 
                            right=0.967, 
                            top=0.913,
                            wspace=0.164, 
                            hspace=0.34)                 
        plt.savefig('transfer.pdf', dpi=300)
        plt.show()

    def _plot(self, index_h, index_w):
        index = index_h * self.w + index_w
        key_scenario = self.data_keys[index]
        scenario_results = self.data[key_scenario]

        print("================================")
        print(key_scenario)
        for key_scenario_result in scenario_results:
            scenario_result_min = self.smooth(np.min(scenario_results[key_scenario_result], axis=0), 0.5)
            scenario_result_max = self.smooth(np.max(scenario_results[key_scenario_result], axis=0), 0.5)

            xdata = np.arange(scenario_result_min.shape[0]) * 1e4

            if self.h == 1:
                self.ax[index_w].fill_between(xdata, 
                                              scenario_result_min, 
                                              scenario_result_max,
                                              where=scenario_result_max>scenario_result_min,
                                              facecolor=self.colors[self.algos[key_scenario_result]],
                                              alpha=0.3)
                line = self.ax[index_w].plot(xdata, 
                                             np.median(scenario_results[key_scenario_result], axis=0),
                                             label=key_scenario_result,
                                             linewidth=2.5,
                                             c=self.colors[self.algos[key_scenario_result]])[:]
            else:
                self.ax[index_h][index_w].fill_between(xdata, 
                                                    scenario_result_min, 
                                                    scenario_result_max,
                                                    where=scenario_result_max>scenario_result_min,
                                                    facecolor=self.colors[self.algos[key_scenario_result]],
                                                    alpha=0.3)
                
                line = self.ax[index_h][index_w].plot(xdata, 
                                                    np.median(scenario_results[key_scenario_result], axis=0),
                                                    label=key_scenario_result,
                                                    linewidth=2.5,
                                                    c=self.colors[self.algos[key_scenario_result]])[:]
                print(key_scenario, key_scenario_result, np.median(scenario_results[key_scenario_result][:, 0]), np.median(scenario_results[key_scenario_result], axis=0)[-5:].mean())
            self.lines[key_scenario_result] = line[0]

        if self.h == 1:
            self.ax[index_w].set_title('('+chr(index+97) + ')' + ' ' +key_scenario, fontsize=16)
            self.ax[index_w].grid(True, linewidth = "0.3")
            self.ax[index_w].set_xlabel('Time Steps', fontsize=16)
            self.ax[index_w].set_ylabel('Test Win % ', fontsize=16)
        else:
            self.ax[index_h][index_w].set_title('('+chr(index+97) + ')' + ' ' +key_scenario, fontsize=16)
            self.ax[index_h][index_w].grid(True, linewidth = "0.3")
            self.ax[index_h][index_w].set_xlabel('Time Steps', fontsize=16)
            self.ax[index_h][index_w].set_ylabel('Test Win % ', fontsize=16)

    @staticmethod
    def get_csv_path(prefix_path):
        try:
            filenames = os.listdir(prefix_path)
            fullnames = [prefix_path + filename for filename in filenames if filename.endswith('.csv')]
            
            return fullnames
        except:
            return None

    @staticmethod
    def read_csv(path_csv):
        try:
            csv_data = pd.read_csv(path_csv)
            value_data = np.array(csv_data['Value'].array)

            return value_data
        except:
            print(f"open csv {path_csv} error")
            return None
            #raise Exception(f"open csv {path_csv} error")

    def smooth(self, data, weight=None):
        smoothed = []
        last = data[0]
        weight = weight if weight else self.weight
        for d in data:
            smooth_val = last * weight + (1 - weight) * d
            smoothed.append(smooth_val)
            last = smooth_val
        
        return np.array(smoothed)


if __name__ == "__main__":
    # 8m9m-5m6m
    path_inverse_5m6m_updet_qmix = './transfer/UPDeT+QMIX/5m6m(from 8m9m)/'
    path_inverse_5m6m_pit_qmix = './transfer/PIT+QMIX/5m6m(from 8m9m)/'
    path_inverse_5m6m_pit_tf = './transfer/PIT+LA-QTransformer(hybrid)/5m6m(from 8m9m)/'
    path_inverse_5m6m_pit_v0_tf = './transfer/PIT+LA-QTransformer(hard)/5m6m(from 8m9m)/'

    paths_inverse_5m6m = [
                    path_inverse_5m6m_updet_qmix,
                    path_inverse_5m6m_pit_qmix,
                    path_inverse_5m6m_pit_tf,
                    path_inverse_5m6m_pit_v0_tf,
                    ]
    
    # 5m6m-8m9m
    path_inverse_8m9m_updet_qmix = './transfer/UPDeT+QMIX/8m9m(from 5m6m)/'
    path_inverse_8m9m_pit_qmix = './transfer/PIT+QMIX/8m9m(from 5m6m)/'
    path_inverse_8m9m_pit_tf = './transfer/PIT+LA-QTransformer(hybrid)/8m9m(from 5m6m)/'
    path_inverse_8m9m_pit_v0_tf = './transfer/PIT+LA-QTransformer(hard)/8m9m(from 5m6m)/'

    paths_inverse_8m9m = [
                    path_inverse_8m9m_updet_qmix,
                    path_inverse_8m9m_pit_qmix,
                    path_inverse_8m9m_pit_tf,
                    path_inverse_8m9m_pit_v0_tf,
                    ]

    # 2s3z-1s2z
    path_inverse_1s2z_updet_qmix = './transfer/UPDeT+QMIX/1s2z(from 2s3z)/'
    path_inverse_1s2z_pit_qmix = './transfer/PIT+QMIX/1s2z(from 2s3z)/'
    path_inverse_1s2z_pit_tf = './transfer/PIT+LA-QTransformer(hybrid)/1s2z(from 2s3z)/'
    path_inverse_1s2z_pit_v0_tf = './transfer/PIT+LA-QTransformer(hard)/1s2z(from 2s3z)/'

    paths_inverse_1s2z = [
                    path_inverse_1s2z_updet_qmix,
                    path_inverse_1s2z_pit_qmix,
                    path_inverse_1s2z_pit_tf,
                    path_inverse_1s2z_pit_v0_tf
                    ]
    
    # 1s2z-2s3z
    path_inverse_2s3z_updet_qmix = './transfer/UPDeT+QMIX/2s3z(from 1s2z)/'
    path_inverse_2s3z_pit_qmix = './transfer/PIT+QMIX/2s3z(from 1s2z)/'
    path_inverse_2s3z_pit_tf = './transfer/PIT+LA-QTransformer(hybrid)/2s3z(from 1s2z)/'
    path_inverse_2s3z_pit_v0_tf = './transfer/PIT+LA-QTransformer(hard)/2s3z(from 1s2z)/'

    paths_inverse_2s3z = [
                    path_inverse_2s3z_updet_qmix,
                    path_inverse_2s3z_pit_qmix,
                    path_inverse_2s3z_pit_tf,
                    path_inverse_2s3z_pit_v0_tf
                    ]
    
    # 3m-5m
    path_inverse_5m_updet_qmix = './transfer/UPDeT+QMIX/5m(from 3m)/'
    path_inverse_5m_pit_qmix = './transfer/PIT+QMIX/5m(from 3m)/'
    path_inverse_5m_pit_tf = './transfer/PIT+LA-QTransformer(hybrid)/5m(from 3m)/'
    path_inverse_5m_pit_v0_tf = './transfer/PIT+LA-QTransformer(hard)/5m(from 3m)/'

    paths_inverse_5m = [
                    path_inverse_5m_updet_qmix,
                    path_inverse_5m_pit_qmix,
                    path_inverse_5m_pit_tf,
                    path_inverse_5m_pit_v0_tf
                    ]
    
    # 3m-7m
    path_inverse_7m_updet_qmix = './transfer/UPDeT+QMIX/7m(from 3m)/'
    path_inverse_7m_pit_qmix = './transfer/PIT+QMIX/7m(from 3m)/'
    path_inverse_7m_pit_tf = './transfer/PIT+LA-QTransformer(hybrid)/7m(from 3m)/'
    path_inverse_7m_pit_v0_tf = './transfer/PIT+LA-QTransformer(hard)/7m(from 3m)/'

    paths_inverse_7m = [
                    path_inverse_7m_updet_qmix,
                    path_inverse_7m_pit_qmix,
                    path_inverse_7m_pit_tf,
                    path_inverse_7m_pit_v0_tf
                    ]
    
    ploter = MultiPlot([
                    paths_inverse_5m,
                    paths_inverse_7m,
                    paths_inverse_5m6m, 
                    paths_inverse_8m9m,
                    paths_inverse_1s2z, 
                    paths_inverse_2s3z,            
                        ])
    
    ploter.plot()