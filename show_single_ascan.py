# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 16:56:23 2021

@author: mzly903

View a single speroid

"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

path = r'C:\Users\mzly903\Downloads'

spheroid = r'\8p9_0p156_Ch11-12_80000_win199-002.mat'

data = pd.read_csv(path+spheroid)

spectra = np.loadtxt(path+spheroid)

ascan_number = 124 # enter Ascan you want to see

fft = np.fft.fft(spectra[ascan_number])
    
ascan = abs(fft)

plt.figure()
plt.plot(ascan)
plt.title(spheroid[1:])
plt.ylabel('Intensity')
plt.xlabel('Depth')
