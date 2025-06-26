#!/user/bin/env python3
# -*- coding: utf-8 -*-
#%%
import os
import tempfile

# 设置joblib的临时文件夹为一个只包含ASCII字符的路径
os.environ['JOBLIB_TEMP_FOLDER'] = os.path.join(tempfile.gettempdir(), 'joblib_temp')

import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')
from mne import find_events, Epochs, compute_covariance, make_ad_hoc_cov

from mne.simulation import (simulate_sparse_stc, simulate_raw,
                            add_noise, add_ecg, add_eog)
import scipy.io
from mne import transforms

import math
from nilearn.plotting import plot_stat_map
from nilearn.image import index_img
from mne.minimum_norm import make_inverse_operator, apply_inverse
from scipy import linalg
import os
from mne.datasets import sample
import copy
#import DSSP
from mne import fixes
from scipy import linalg


#%%empty_room1生成N2C
############################################################
###数据加载
Empty1file = r'E:\study\MEG\MEG\data\empty_room2.mat'
Empty1_from_file = scipy.io.loadmat(Empty1file)
Emptyroom1data = np.array(Empty1_from_file["data_orig"])#加入仿真信号中的噪声数据

Rawdata = np.zeros((27,300000),dtype=np.double)
#Rawdata = np.array(Raw_from_file["B1"])    # (730048, 32)


###加载数据通道信息
chan0_pick = np.arange(27)
#chan0_pick = [0, 1, 2, 3, 4,  6, 7, 8, 9, 10, ]  # 10cm---03.03

raw_chose = Rawdata[chan0_pick]  # 选择通道，截取片段
#raw_chose = raw_chose / 2.7 / 1000000000 / 0.33  # 电压V转化为特斯拉T
# Raw1的第31行是什么？刺激通道吗？为什么要除以3？ 增益的问题
#raw0 = np.row_stack((raw_chose, Rawdata[30] / 3))  # 行合并,除以3，电压为1，事件id为1
raw0 = raw_chose
a = 0
b = 300000
#raw0 = raw_chose
chan1_pick = [0, 1, 2, 3,    7,  9,
              10,  12,  14, 15,  17, 18, 19,
               21,   24, 25, 26, 27, 28, 29,
              30, 31, 32, 33, 34, 35]#左手30

Emptyroom1_raw_chose = Emptyroom1data[chan1_pick,]
Emptyroom1_raw_chose = Emptyroom1_raw_chose / 2.7 / 1e9 / 0.33
# 确保两个数组在连接轴上具有相同的维度

# 获取最大长度
max_length = max(Emptyroom1_raw_chose.shape[1], raw0[26].shape[0])

# 使用边缘值填充
Emptyroom1_raw_chose_padded = np.pad(Emptyroom1_raw_chose, ((0, 0), (0, max_length - Emptyroom1_raw_chose.shape[1])), mode='edge')
raw0_26_padded = np.pad(raw0[26], (0, max_length - raw0[26].shape[0]), mode='edge')
Emptyroom1_raw0 = np.row_stack((Emptyroom1_raw_chose_padded, raw0_26_padded))


info = mne.create_info(
    ch_names=[f'{n}' for n in range(0,26)]+['STI 014'],


    ch_types=['eeg']*26+['stim'],
    sfreq=1000)
Raw0 = mne.io.RawArray(raw0, info)#真实数据
#event0 = mne.find_events(Raw0, stim_channel='STI 014', initial_event=True)

# 加载去噪后的拼接数据并替换Raw0中的信号数据
denoised_file = r'F:\home\homeForStudy\N2C\EX4\visualization_results\concatenated_denoised.mat'
if os.path.exists(denoised_file):
    print(f"正在加载去噪数据: {denoised_file}")
    denoised_data_mat = scipy.io.loadmat(denoised_file)
    denoised_data = denoised_data_mat['data']  # 形状应该是 (26, time_points)
    
    # 确保去噪数据有正确的形状
    if denoised_data.shape[0] == 26:
        print(f"去噪数据形状: {denoised_data.shape}")
        
        # 创建新的数据数组，包含去噪信号和刺激通道
        # 前26个通道使用去噪数据，最后一个通道保持原来的刺激通道
        new_raw_data = np.zeros((27, denoised_data.shape[1]))
        new_raw_data[0:26, :] = denoised_data  # 前26个通道使用去噪数据
        new_raw_data[26, :] = raw0[26, :denoised_data.shape[1]] if raw0.shape[1] >= denoised_data.shape[1] else np.pad(raw0[26], (0, denoised_data.shape[1] - raw0.shape[1]), mode='constant')
        
        # 重新创建Raw对象
        Raw0 = mne.io.RawArray(new_raw_data, info)
        print("成功将去噪数据替换到Raw0中")
    else:
        print(f"警告: 去噪数据通道数({denoised_data.shape[0]})与期望的26通道不匹配，使用原始零数据")
else:
    print(f"警告: 找不到去噪数据文件 {denoised_file}，使用原始零数据")

Emptyroom1_Raw0 = mne.io.RawArray(Emptyroom1_raw0, info)#真实空房间数据

sensorfile = scipy.io.loadmat(r'E:\study\MEG\MEG\data\20230726zrcpos')  # E://mne//26channel-8.31//数据//pos_cfz.mat
sensor_data = sensorfile['new_sensors']
sensor_pos = sensor_data[0, 0]['pos'] # (104, 3) 这是什么坐标？？
sensor_pos = sensor_pos.T / 1000  # （3, 104)
sensor_pos.shape[0]
sensor_ori = sensor_data[0,0]['ori']
sensor_ori = sensor_ori
sensor_ori = sensor_ori.T
# j是什么？
sensor_pick = [9,18,29,5,8,27,22,17,19,30,15,11,10,7,3,0,26,20,4,24,2,31,21,23,25,13]
pos_final = sensor_pos[0:3,sensor_pick]#提取26个通道的传感器位置
ori_final=sensor_ori[0:3,sensor_pick]
dic = {str(i):pos_final[0:3,i] for i in range(26)}  #26!
montage = mne.channels.make_dig_montage(ch_pos=dic,coord_frame='head')
Raw0 = Raw0.set_montage(montage)
Emptyroom1_Raw0 = Emptyroom1_Raw0.set_montage(montage)

montage.plot(kind='topomap', show_names=True)
plt.show(block=True)
for j in range(26):
    Raw0.info['dig'][j]['kind'] = 4
    Raw0.info['chs'][j]['kind'] = mne.io.constants.FIFF.FIFFV_MEG_CH
    Raw0.info['chs'][j]['unit'] = mne.io.constants.FIFF.FIFF_UNIT_T
    Raw0.info['chs'][j]['coil_type'] = mne.io.constants.FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2
    Raw0.info['chs'][j]['loc'][3:12] = np.array([1., 0., 0., 0., 1., 0., 0., 0., 1.])
    Z_orient = mne._fiff.tag._loc_to_coil_trans(Raw0.info['chs'][j]['loc'])[:3, :3]
    find_Rotation = transforms._find_vector_rotation(Z_orient[:, 2], ori_final[:, j])
    #Raw0.info['chs'][j]['loc'][3:12] = np.dot(find_Rotation, Z_orient).T.ravel()
    Emptyroom1_Raw0.info['dig'][j]['kind'] = 4
    Emptyroom1_Raw0.info['chs'][j]['kind'] = mne.io.constants.FIFF.FIFFV_MEG_CH
    Emptyroom1_Raw0.info['chs'][j]['unit'] = mne.io.constants.FIFF.FIFF_UNIT_T
    Emptyroom1_Raw0.info['chs'][j]['coil_type'] = mne.io.constants.FIFF.FIFFV_COIL_QUSPIN_ZFOPM_MAG2
    Emptyroom1_Raw0.info['chs'][j]['loc'][3:12] = np.array([1., 0., 0., 0., 1., 0., 0., 0., 1.])
    Z_orient = mne._fiff.tag._loc_to_coil_trans(Emptyroom1_Raw0.info['chs'][j]['loc'])[:3, :3]
    find_Rotation = transforms._find_vector_rotation(Z_orient[:, 2], ori_final[:, j])
    #Emptyroom1_Raw0.info['chs'][j]['loc'][3:12] = np.dot(find_Rotation, Z_orient).T.ravel()
Emptyroom1_Raw0.plot_psd(fmax=100,picks='meg',average=False,color='black',area_mode=None,show=True)
plt.show(block=True)


# %%


Raw1 = Raw0.copy()
Raw1 = Raw1.filter(l_freq=1, h_freq=40)
Raw2 = Raw1.copy()

#%%

events_raw2 = mne.make_fixed_length_events(Raw2, id=1, duration=1.0)
epoch_raw2 = mne.Epochs(Raw2, events_raw2, 1, tmin=-0.2, tmax=1.0, preload=True)
evoked_raw2 = epoch_raw2.average()
evoked_raw2.plot()
plt.show(block=True)