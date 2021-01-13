# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:28:30 2021

@author: mzly903
"""

import numpy as np
from matplotlib import pyplot as plt
import OCTFunctions as octf

import scipy.stats as stats # gaussian
# plt.close('all')
path = r'C:\Users\mzly903\Desktop\PhD\2. Data\1. Spheroids\Experiment 4 21 Jan 2017\21JanMeasurements'

spheroid = r'\b1_003.txt'
def normalize(data):
    return [x/max(data) for x in data]

def apply_gauss_window(data = [1, 2], cw = 770, sigma = 15, wave_from = 740, wave__to = 980 ):
    mu = cw
    x = np.linspace(wave_from, wave__to, len(data))
    gaussian_distribution = normalize(stats.norm.pdf(x, mu, sigma))
    after_gaussian_spectrum = data*gaussian_distribution
    return after_gaussian_spectrum



zero = 18
data = np.loadtxt(path+spheroid)


image = octf.bscan(data, 0, len(data), zepopadding = 14)
octf.showme(image)
number = int(plt.ginput(1)[0][0])

# plt.close('all')
# plt.figure()
# number = int(input('enter ascan Number'))

from mpl_toolkits import mplot3d 
import numpy as np 
import matplotlib.pyplot as plt 
# fig = plt.figure() 
# ax = plt.axes(projection ='3d')
peaks = [] 
for i in range(8):
    
    
    wave = 810+i*5
    
    modified_data = apply_gauss_window(data[number], wave, 15, 760, 920)
    
    ascan= abs(np.fft.fft(modified_data, n=2**zero))
    
    crop = ascan[11700:12800]#[36500:38500]
    
    x = range(len(crop))
    print(i)
    
    plt.figure()
    plt.plot(modified_data)
    plt.title('cw: ' + str(wave))
    # plt.plot(crop)
    # plt.text(i*(len(crop)/30), max(crop), str(wave))
    # a = plt.ginput(1)
    

    
    # ax.plot3D(x, [wave]*len(crop), crop)
    # ax.plot3D([np.argmax(crop)]*100, [wave]*100, np.linspace(0, max(crop), 100))
    
    peaks.append(np.argmax(crop))

plt.figure()
plt.plot(peaks)    
plt.xlabel('depth')
plt.ylabel('wavelenth [nm]')
ax.set_zlabel('intensity')
ax.set_title('Spheroid ' + spheroid[1:]) 
plt.show() 