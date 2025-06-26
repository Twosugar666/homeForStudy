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
Empty1file = r'E:\study\MEG\MEG\data\empty_room1.mat'
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
#%%###加载内部仿真数据

data_path = r'E:\study\MEG\MEG\data\MNE-sample-data'
subjects_dir = data_path + '/subjects'
subject = 'zhaoruochen'
#trans =  r'D:\study\MEG\MEG\MNE-sample-data\subjects\wuhuanqi\whq_mri-trans.fif'
#trans =mne.read_trans(trans)
trans = mne.transforms.Transform('head', 'mri')
surface = r'E:\study\MEG\MEG\data\MNE-sample-data\subjects\zhaoruochen\bem/inner_skull.surf'
mri = r'"E:\study\MEG\MEG\data\MNE-sample-data\subjects\zhaoruochen\mri\T1.mgz"'
#src = mne.setup_source_space(subject, spacing='oct6', add_dist='patch',
                             #subjects_dir=subjects_dir)
#conductivity = (0.3,0.006, 0.3)
conductivity = (0.3,)
model_sim = mne.make_bem_model(subject='zhaoruochen',
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem_sim = mne.make_bem_solution(model_sim)

src_pos_inner = {}
src_pos_inner['nn'] = np.array([1, 1, 1])#内部仿真源方向
#src_pos_inner['rr'] = np.array([0.0454, -0.0265, 0.0635])#内部仿真源位置
src_pos_inner['rr'] = np.array([0.0414,-0.0133,0.0666])#内部仿真源位置8.032906532	-14.01068687	91.4282150341.40, -13.34, 66.63

src_pos_inner['nn'] = copy.copy(src_pos_inner['nn'][np.newaxis, :])
src_pos_inner['rr'] = copy.copy(src_pos_inner['rr'][np.newaxis, :])

vol_src = mne.setup_volume_source_space(subject,pos=src_pos_inner, bem=bem_sim )
#src =  vol_src
  # for single layer
# conductivity = (0.3, 0.006, 0.3)  # for three layers

fwd_sim = mne.make_forward_solution(Raw0.info, trans=trans, src=vol_src, bem=bem_sim,
                                meg=True, eeg=False, mindist=5.0, n_jobs=1,
                                verbose=True)
src = fwd_sim['src']
mne.viz.plot_alignment(Raw0.info, trans=trans, subject='zhaoruochen',
                       src=src, subjects_dir=subjects_dir, dig=False,
                       surfaces=[ 'white'], coord_frame='mri')
#fig = mne.viz.plot_alignment(Raw0.info,trans=trans,dig=True,eeg=False,surfaces=[],meg = ['helmet'],coord_frame='meg',src = vol_src)
n_dipoles = 1  # number of dipoles to create
epoch_duration = 1  # duration of each epoch/event
n = 0  # harmonic number
rng = np.random.RandomState(0)  # random state (make reproducible)


def data_fun(times):
        """Generate time-staggered sinusoids at harmonics of 10Hz"""
        global n
        n_samp = len(times)
        window = np.zeros(n_samp)
        start, stop = [int(ii * float(n_samp) / (2 * n_dipoles))
                       for ii in (2 * n, 2 * n + 1)]
        window[start:stop] = 1.
        n += 1
        data = 5e-8 * np.sin(2. * np.pi * (10.) * n * times)
        data *= window
        return data


times = Raw0.times[:int(Raw0.info['sfreq'] * epoch_duration)]
stc_sim = simulate_sparse_stc(src, n_dipoles=n_dipoles, times=times,
                          data_fun=data_fun, random_state=rng)
# look at our source data
fig, ax = plt.subplots(1)
ax.plot(times, 1e9 * stc_sim.data.T)
ax.set(ylabel='Amplitude (nAm)', xlabel='Time (sec)')
mne.viz.utils.plt_show()
plt.show(block=True)

simsignal_Raw = simulate_raw(Raw0.info, [stc_sim] * 300, forward=fwd_sim, verbose=True)#仿真信号数据
#cov = make_ad_hoc_cov(simsignal_Raw.info)
#cov['data']*=0.25e-22
#add_noise(simsignal_Raw, cov, iir_filter=[0.2, -0.2, 0.04], random_state=rng)

simsignal_Raw.plot()
events_simin = find_events(simsignal_Raw)
epoch_simin = mne.Epochs(simsignal_Raw, events_simin,  1,baseline=(-0.2,0),tmin=-0.2, tmax=epoch_duration,)
evoked_simin = epoch_simin.average()
fig, ax = plt.subplots(1,)
fig = evoked_simin.plot(axes=ax,spatial_colors=True,titles=None)
for text in list(ax.texts):
    text.remove()
#ax.grid(True)
ax.grid(axis="y",linestyle='--')
ax.set_yticks([-800,-600,-400,-200, 0, 200,400,600,800] )
#fig.savefig(r'D:\zrc\博士\课题\teHFC\论文图片\半仿真\evoked_simin.tiff',dpi=300)
plt.show(block=True)

#%%###仿真源合并

simsignal_raw_data,_ = simsignal_Raw[:]
#outersignal_raw_data,_ = outersignal_Raw0[:]
#junyun_raw_data,_ = junyun_Raw0[:]
# 裁剪Emptyroom1_raw0以匹配simsignal_raw_data的形状
Emptyroom1_raw0_data = Emptyroom1_raw0[:, :simsignal_raw_data.shape[1]]
finaldata = simsignal_raw_data + Emptyroom1_raw0_data #仿真信号+空房间噪声
finalsim_Raw0 = mne.io.RawArray(finaldata, Raw0.info)

#%%带通
raw01 = finalsim_Raw0.copy()
finalsim_Raw1 = raw01.filter(l_freq=1, h_freq=40)
Emptyroom1_Raw01 = Emptyroom1_Raw0.copy()
Emptyroom1_Raw1 = Emptyroom1_Raw01.filter(l_freq=1, h_freq=40)
Emptyroom2_Raw1 = Emptyroom1_Raw1.copy()

#%%

events_emptyroom1 = mne.make_fixed_length_events(Emptyroom1_Raw0, id=1, duration=1.0)
epoch_emptyroom1 = mne.Epochs(Emptyroom1_Raw0, events_emptyroom1, 1, tmin=-0.2, tmax=1.0, preload=True)
evoked_emptyroom1 = epoch_emptyroom1.average()
evoked_emptyroom1.plot()
plt.show(block=True)

#%%empty_room1_data
events_bandpass = mne.make_fixed_length_events(simsignal_Raw, id=1, duration=1.0)
epoch_bandpass= mne.Epochs(finalsim_Raw1, events_bandpass,  1, tmin=-0.2, baseline=(None, 0),detrend=1,tmax=1)#
bandpass_epoch = epoch_bandpass.get_data(picks=['meg'])
evoked_bandpass = epoch_bandpass.average()
evoked_bandpass.plot(spatial_colors=True,titles=None)
plt.show(block=True)

bandpass_evoked = evoked_bandpass.get_data(picks=['meg'])#Y
simin_evoked=evoked_simin.get_data(picks=['meg'])#Y
evoked_simin.plot(spatial_colors=True,titles=None)
plt.show(block=True)
#%%empty_room1_data_mat生成N2C
# 假设 finalsim_Raw1 是带通滤波后的 RawArray
out_fname = 'finalsim_raw-1-40hz_raw.fif'
finalsim_Raw1.save(out_fname, overwrite=True)


#%%whq_data_mat生成N2C
import numpy as np
import scipy.io as sio
import os
selected_epochs_data = bandpass_epoch[:, :, 200:1201]  # 形状变为 (298, 28, 1001)
selected_evoked_data = simin_evoked[:, 200:1201]  # 形状变为 (28, 1001)

# 创建目录（如果不存在）
os.makedirs('traindata_eptr1/noisy', exist_ok=True)
os.makedirs('traindata_eptr1/clean', exist_ok=True)

# 文件编号
file_index = 1

# 遍历所有epoch
for j in range(selected_epochs_data.shape[0]):  # 遍历所有epochs
    # 对于每个epoch，保存所有通道的带噪声数据
    noisy_filename = f'traindata_eptr1/noisy/noisy_{file_index}.mat'
    sio.savemat(noisy_filename, {'data': selected_epochs_data[j].astype(np.float64)})  # 保存形状为(28, 1001)的数据

    # 对于每个epoch，保存所有通道的干净数据
    clean_filename = f'traindata_eptr1/clean/clean_{file_index}.mat'
    sio.savemat(clean_filename, {'data': selected_evoked_data.astype(np.float64)})  # 保存形状为(28, 1001)的数据

    # 更新文件编号
    file_index += 1

print("所有文件已成功保存。")

# %%
