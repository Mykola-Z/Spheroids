# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 16:56:23 2021

@author: mzly903

View a single speroid

"""

import numpy as np
from matplotlib import pyplot as plt

path = r'C:\Users\mzly903\Desktop\PhD\2. Data\1. Spheroids\Experiment 4 21 Jan 2017\21JanMeasurements'

spheroid = r'\a1_008.txt'

spectra = np.loadtxt(path+spheroid)

ascan_number = 124 # enter Ascan you want to see

fft = np.fft.fft(spectra[ascan_number])
    
ascan = abs(fft)

plt.figure()
plt.plot(ascan)
plt.title(spheroid[1:])
plt.ylabel('Intensity')
plt.xlabel('Depth')
