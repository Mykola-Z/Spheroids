# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 16:56:23 2021

@author: mzly903

View single speroid

"""

import numpy as np
from matplotlib import pyplot as plt

path = r'C:\Users\mzly903\Desktop\PhD\2. Data\1. Spheroids\Experiment 4 21 Jan 2017\21JanMeasurements'

spheroid = r'\a1_008.txt'

spectra = np.loadtxt(path+spheroid)

bscan = []

for i in range(len(spectra)):
    
    fft = np.fft.fft(spectra[i])
    
    ascan = abs(fft)
    
    bscan.append(ascan)

bscan = np.rot90(bscan)
plt.figure()
plt.imshow(bscan, cmap = 'gray')
plt.clim([500, 7000])
plt.title(spheroid[1:])
plt.ylabel('Depth')
plt.xlabel('Width')
plt.axis('tight')
