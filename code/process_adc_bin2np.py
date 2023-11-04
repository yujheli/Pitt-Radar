import numpy as np
import cv2
import os

import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='Process the binary file to adc',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--raw_data', metavar='DIR', default='./radar_dataset_bin/',
                    help='path to data folder')
parser.add_argument('--output', '-o', metavar='DIR', default='./radar_dataset_numpy/',
                    help='path to output folder. If not set, will be created in data folder')

args = parser.parse_args()


if not os.path.exists(args.output):
        os.makedirs(args.output)

# range_list = range(step*args.ID,step*args.ID+step)

numADCSamples = 128
numTxAntennas = 3
numRxAntennas = 4
numLoopsPerFrame = 128
numChirpsPerFrame = numTxAntennas * numLoopsPerFrame
num_angle_bins = 128

numADCSamples = 256
num_angle_bins = 256


sizes = (128,128)
fig = plt.figure()
fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward=False)

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()

fig.add_axes(ax)

def RAD_map(range_plot):
    range_doppler = np.fft.fft(range_plot, axis=0)
    range_doppler = np.fft.fftshift(range_doppler, axes=0)

    padding = ((0,0), (0,num_angle_bins-range_doppler.shape[1]), (0,0))
    range_azimuth = np.pad(range_doppler, padding, mode='constant')
    range_azimuth = np.fft.fft(range_azimuth, axis=1)
    range_azimuth = np.fft.fftshift(range_azimuth, axes=1)
    
    # out_img = (np.abs(range_azimuth).sum(0).T)
    # out_img = np.log2(np.abs(range_azimuth)) # Maintain the cube
    out_img = np.abs(range_azimuth) # Maintain the cube


    # out_img = np.flip(out_img,axis=2).transpose(2, 1, 0)
    out_img = np.rot90(out_img,2,axes=(1,2)).transpose(2, 1, 0)

    return out_img

    
dca = DCA1000(static_ip='127.0.0.1',data_port=6666)
files = os.listdir(args.raw_data)
for i in range(len(files)):
    # print(i)
    filename = args.raw_data+'{:06d}.bin'.format(i)
    filename = args.raw_data+'frame_{:06d}.bin'.format(i)
    # adc_data = np.fromfile(fileName, dtype=np.uint16)

    adc_data = np.fromfile(filename, dtype=np.uint16)
        
    adc_data2frame = dca.organize(adc_data, num_chirps=numChirpsPerFrame, num_rx=numRxAntennas, num_samples=numADCSamples)

    # out_img, file = process_radar(file=adc_data2frame)
    file = adc_data2frame
    num_tx=3
    vx_axis=1

    file.real = file.real.astype(np.int16)
    file.imag = file.imag.astype(np.int16)

    file_adc = np.concatenate([file[i::num_tx, ...] for i in range(num_tx)], axis=vx_axis)
    np.save(args.output+'{:d}'.format(i),file_adc)


    
    







