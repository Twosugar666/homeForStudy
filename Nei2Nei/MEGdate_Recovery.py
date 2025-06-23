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
from pathlib import Path
#%%
############################################################
###数据加载


Empty1file = 'E:\study\MEG\MEG\data\whq_kfj.mat'
Empty1_from_file = scipy.io.loadmat(Empty1file)
Emptyroom1data = np.array(Empty1_from_file["B1"])#加入仿真信号中的噪声数据

#Rawdata = np.zeros((32,300000),dtype=np.double)
Rawdata = np.zeros((29,300000),dtype=np.double)#6.18 N2C 28channal
#Rawdata = np.array(Raw_from_file["B1"])    # (730048, 32)


###加载数据通道信息
#chan0_pick = np.arange(32)
chan0_pick = np.arange(29)#6.18 N2C 28channal
#chan0_pick = [0, 1, 2, 3, 4,  6, 7, 8, 9, 10, ]  # 10cm---03.03

raw_chose = Rawdata[chan0_pick]  # 选择通道，截取片段
#raw_chose = raw_chose / 2.7 / 1000000000 / 0.33  # 电压V转化为特斯拉T
# Raw1的第31行是什么？刺激通道吗？为什么要除以3？ 增益的问题
#raw0 = np.row_stack((raw_chose, Rawdata[30] / 3))  # 行合并,除以3，电压为1，事件id为1
raw0 = raw_chose
a = 0
b = 300000
#raw0 = raw_chose
#chan1_pick = [0, 1, 2, 3, 4,  6, 7, 8, 9, 10,  12, 13, 14, 15, 16,
     #17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,  29, 31, 32, 33, 34]#teHFC
chan1_pick = [0, 1, 2, 3,   6, 7, 8, 9, 10,  12, 13, 14, 15, 16,
     17, 18, 19, 20, 21, 22, 23, 24, 25,  27,  29, 31,  33, 34]#N2C 28chan,4\26\32

Emptyroom1_raw_chose = Emptyroom1data[chan1_pick,a:b]
Emptyroom1_raw_chose = Emptyroom1_raw_chose / 2.7 / 1e9 / 0.33
Emptyroom1_raw0 = np.row_stack((Emptyroom1_raw_chose, raw0[28]))

info = mne.create_info(
    ch_names=[f'{n}' for n in range(0,28)]+['STI 014'],


    ch_types=['eeg']*28+['stim'],
    sfreq=1000)
Raw0 = mne.io.RawArray(raw0, info)#真实数据
#event0 = mne.find_events(Raw0, stim_channel='STI 014', initial_event=True)

Emptyroom1_Raw0 = mne.io.RawArray(Emptyroom1_raw0, info)#真实空房间数据

sensorfile = scipy.io.loadmat('E:\study\MEG\MEG\data\wuhuanqipos.mat')  # E://mne//26channel-8.31//数据//pos_cfz.mat
sensor_data = sensorfile['new1']
sensor_pos = sensor_data[0, 0]['pos'] # (104, 3) 这是什么坐标？？
sensor_pos = sensor_pos.T / 1000  # （3, 104)
sensor_pos.shape[0]
sensor_ori = sensor_data[0,0]['ori']
sensor_ori = sensor_ori
sensor_ori = sensor_ori.T
# j是什么？
#sensor_pick = [83, 77, 73, 18, 7,  1, 54, 0, 16, 81,  55, 56, 53, 80, 84, 64, 8, 67, 9, 66,
     #82, 75, 29, 65, 3, 74,  27, 10, 17, 76, 6]#teHFC
sensor_pick = [83, 77, 73, 18,   1, 54, 0, 16, 81,  55, 56, 53, 80, 84, 64, 8, 67, 9, 66,
     82, 75, 29, 65,  74,  27, 10,  76, 6]#N2C 28chan,7\3\17
pos_final = sensor_pos[0:3,sensor_pick]#提取28个通道的传感器位置  j!
ori_final=sensor_ori[0:3,sensor_pick]
dic = {str(i):pos_final[0:3,i] for i in range(28)}  #26!
montage = mne.channels.make_dig_montage(ch_pos=dic,coord_frame='head')
Raw0 = Raw0.set_montage(montage)
Emptyroom1_Raw0 = Emptyroom1_Raw0.set_montage(montage)
montage.plot(kind='topomap', show_names=True)
plt.show(block=True)
for j in range(28):
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
finaldata = simsignal_raw_data+Emptyroom1_raw0 #仿真信号+空房间噪声
finalsim_Raw0 = mne.io.RawArray(finaldata, Raw0.info)

#%%带通
raw01 = finalsim_Raw0.copy()
finalsim_Raw1 = raw01.filter(l_freq=1, h_freq=40)
Emptyroom1_Raw01 = Emptyroom1_Raw0.copy()
Emptyroom1_Raw1 = Emptyroom1_Raw01.filter(l_freq=1, h_freq=40)
Emptyroom2_Raw1 = Emptyroom1_Raw1.copy()


#%%whq_data
events_bandpass = find_events(simsignal_Raw)
epoch_bandpass= mne.Epochs(finalsim_Raw1, events_bandpass,  1, tmin=-0.2, baseline=(None, 0),detrend=1,tmax=1)
bandpass_epoch = epoch_bandpass.get_data(picks=['meg'])
evoked_bandpass = epoch_bandpass.average()
bandpass_evoked = evoked_bandpass.get_data(picks=['meg'])#Y
simin_evoked=evoked_simin.get_data(picks=['meg'])#Y
evoked_simin.plot(spatial_colors=True,titles=None)
evoked_bandpass.plot(spatial_colors=True,titles=None)
plt.show(block=True)

import scipy.io as sio

# 先复制一份原始的Epochs对象
epoch_bandpass2 = epoch_bandpass.copy()

# 假设 bandpass_epoch 是你已经处理过的 Epochs 对象
bandpass_epoch2 = epoch_bandpass2.get_data(picks='meg')  # Shape: (epochs, channels, times)

# 获取原始测试文件列表，用于映射索引
test_files = sorted(list(Path('./all_whq/whq_test/noisy').rglob('*.mat')))
print(f"Found {len(test_files)} test files")
print(f"Epoch shape: {bandpass_epoch2.shape}")  # (epochs, channels, times)

# 创建文件名到去噪数据的映射
denoised_data_dict = {}
successful_loads = 0
failed_loads = 0

print("Loading denoised data...")
for test_file in test_files:
    test_file_stem = test_file.stem
    denoised_filename = f'F:/Nei2Nei/denoised_results/{test_file_stem}_denoised.mat'
    
    try:
        # 读取去噪后的数据
        mat_contents = sio.loadmat(denoised_filename)
        # 尝试不同的键名
        if 'data' in mat_contents:
            denoised_data = mat_contents['data'].squeeze()
        elif 'B1' in mat_contents:
            denoised_data = mat_contents['B1'].squeeze()
        else:
            print(f"Warning: Unknown data key in {denoised_filename}")
            failed_loads += 1
            continue
            
        # 确保数据长度正确
        if len(denoised_data) >= 1001:
            denoised_data = denoised_data[:1001]  # 裁剪到1001个时间点
            denoised_data_dict[test_file_stem] = denoised_data * 1e-15  # 转换单位
            successful_loads += 1
        else:
            print(f"Warning: Denoised data length {len(denoised_data)} < 1001 in {denoised_filename}")
            failed_loads += 1
            
    except FileNotFoundError:
        print(f"Warning: File not found: {denoised_filename}")
        failed_loads += 1
        continue
    except Exception as e:
        print(f"Error processing {denoised_filename}: {e}")
        failed_loads += 1
        continue

print(f"Successfully loaded {successful_loads} denoised files, failed: {failed_loads}")



# 方案1: 如果去噪文件是按照 epoch -> channel 的顺序生成的
file_index = 0
for j in range(bandpass_epoch2.shape[0]):  # epochs
    for i in range(bandpass_epoch2.shape[1]):  # channels
        if file_index < len(test_files):
            test_file_stem = test_files[file_index].stem
            if test_file_stem in denoised_data_dict:
                # 替换数据
                bandpass_epoch2[j, i, 200:1201] = denoised_data_dict[test_file_stem]
            file_index += 1
        else:
            break
    if file_index >= len(test_files):
        break

print("Data replacement completed.")

# 计算所有 epoch 的平均值
mean_data2 = np.mean(bandpass_epoch2, axis=0)
mean_data2 =mean_data2[:,200:1201]
bandpass_evoked=evoked_bandpass.get_data(picks=['meg'])#Y
mean_data1 = bandpass_evoked
mean_data1 = mean_data1[:,200:1201]
simin_evoked=evoked_simin.get_data(picks=['meg'])#Y
mean_data3 = simin_evoked
mean_data3 = mean_data3[:,200:1201]
# 创建一个 MNE Evoked 对象
info = evoked_bandpass.info
#evoked2 = mne.EvokedArray(mean_data2, info)
#evoked1 = mne.EvokedArray(mean_data1, info)


# 绘制图形
#evoked.plot(spatial_colors=True, titles='Average Response')

# 如果你想用 matplotlib 自定义更多细节
times = np.linspace(0, 1, 1001)  # 假设 evoked1 和 evoked2 的时间轴是相同的

# 第一张图：展示 mean_data1
fig1, ax1 = plt.subplots()
for ch_idx in range(mean_data1.shape[0]):
    ax1.plot(times, mean_data1[ch_idx], )

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude (AU)')
ax1.set_title('noisy_average evoek')
ax1.set_xticks(np.arange(0, 1.1, 0.1))
fig1.savefig('noisy_average.png', dpi=300, bbox_inches='tight')
plt.show()

# 第二张图：展示 mean_data2
fig2, ax2 = plt.subplots()
for ch_idx in range(mean_data2.shape[0]):
    ax2.plot(times, mean_data2[ch_idx], )

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Amplitude (AU)')
ax2.set_title('denoised_average evoke')
ax2.set_xticks(np.arange(0, 1.1, 0.1))
fig2.savefig('denoised_average.png', dpi=300, bbox_inches='tight')
plt.show()

fig3, ax3 = plt.subplots()
for ch_idx in range(mean_data3.shape[0]):
    ax3.plot(times, mean_data3[ch_idx], )

ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Amplitude (AU)')
ax3.set_title('original_average evoke')
ax3.set_xticks(np.arange(0, 1.1, 0.1))
fig3.savefig('original_average.png', dpi=300, bbox_inches='tight')
plt.show(block=True)