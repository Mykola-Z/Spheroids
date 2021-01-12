# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 14:38:10 2019

@author: mzly903
"""
import glob
import threading
import scipy.constants
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
import peakutils
import scipy
from scipy.ndimage import gaussian_filter
import mahotas as mh
from IPython.html.widgets import interact, fixed
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from contextlib import suppress


filepath2 = r'C:\Users\mzly903\Desktop\PhD\2. Data\BK7\1704\magda'


filepath3 = r'C:\Users\mzly903\Desktop\PhD\2. Data\BK7\1704\magda'
filename_lin_phase = 'lin_st54_len1888'
filename_nonlin_phase = 'nonlin_st54_len1888'
filename_disp = 'disp_st0_len1888'
filename_lin_phase_m = 'lin_st54_len1888'
filename_nonlin_phase_m = 'nonlin_st54_len1888'

cut_point = 105 # Cut point for lin and disp files 

#File uploading 
phase_lin = np.loadtxt(filepath2 + '\\' + filename_lin_phase + '.txt')
phase_nonlin = np.loadtxt(filepath2 + '\\' + filename_nonlin_phase + '.txt')
dispCompVect = np.loadtxt(filepath2 + '\\' + filename_disp + '.txt')
phase_lin_matlab = np.loadtxt(filepath2 + '\\' + filename_lin_phase_m + '.txt')
phase_nonlin_matlab = np.loadtxt(filepath2 + '\\' + filename_nonlin_phase_m + '.txt')

pretty_data = []
sigma = 4
p=15
number_ascans = 190
Zero = 2**p # 2**11=2048 no zero padding
cut_DC =  (2**(p-10))*21
def click():
    root = Tk()
    
    root.withdraw() 
    print ('choose file:')
    name = askopenfilename() 
    print(name + ' is selected')
    spectrum = np.loadtxt(name)
    return spectrum, name

def calibrate(oct_data, disp = True):   # If you want to calibrate 
           
    if isinstance(oct_data[0], (np.ndarray)):
        
        spectrum_lin_disp = []
        
        if disp == True:
            
            for i in range (len(oct_data)):
                
                trim_oct_data = oct_data [i, cut_point:cut_point+len(dispCompVect)]
                f = interp1d(phase_nonlin, trim_oct_data, fill_value='extrapolate')
                spectrum_lin = f(phase_lin)
                spectrum_lin_disp_i =spectrum_lin *np.exp(-1j*dispCompVect)
                spectrum_lin_disp.append(spectrum_lin_disp_i)
                
        elif disp == False:
            
            for i in range (len(oct_data)):
                trim_oct_data = oct_data [i, cut_point:cut_point+len(dispCompVect)]
                f = interp1d(phase_nonlin, trim_oct_data, fill_value='extrapolate')
                spectrum_lin_disp_i = f(phase_lin)
                spectrum_lin_disp.append(spectrum_lin_disp_i)
 
            
    elif isinstance(oct_data[0], (float, int)):
        
        trim_oct_data = oct_data [cut_point:cut_point+len(phase_lin)]
        
        if disp == True:
            
            f = interp1d(phase_nonlin, trim_oct_data, fill_value='extrapolate')
            spectrum_lin = f(phase_lin)
            spectrum_lin_disp =spectrum_lin *np.exp(-1j*dispCompVect)
            
        elif disp == False:
            
            f = interp1d(phase_nonlin, trim_oct_data, fill_value='extrapolate')
            spectrum_lin_disp = f(phase_lin)
        
    return spectrum_lin_disp


def calibrate2(origin, disp = False):
    if disp == False:
        spectrum_data = origin 

        f = interp1d(phase_nonlin_matlab, spectrum_data, fill_value='extrapolate')
        
        spectrum_lin_disp = f(phase_lin_matlab)
        
    else:
        spectrum_data = origin [105:105+len(phase_lin)]
        f = interp1d(phase_nonlin, spectrum_data, fill_value='extrapolate')
        spectrum_lin = f(phase_lin)
        spectrum_lin_disp =spectrum_lin *np.exp(-1j*dispCompVect)
    return spectrum_lin_disp


def ascan(enter_data, p = 15):
    absFFT = np.abs(np.fft.fft(enter_data, n = 2**p))
    return absFFT


def fourrier (data, first_ascan = 0, last_ascan = number_ascans):
    
    if isinstance(data[0], (np.ndarray)) or isinstance(data[0], list):
        data_bscan = []
        
        for i in range (first_ascan, last_ascan):
            absFFT = np.fft.fft(data [i], n=Zero)
            data_bscan.append(absFFT)
            
        result = np.rot90(data_bscan,1)
        
    else:
        
        result = np.fft.fft(data, n = Zero)

    return result
        
 
def bscan(y, first_ascan = 0, last_ascan = number_ascans, zepopadding = 12, go = 0):

    data_bscan = []
    
    
    for g in range (first_ascan, last_ascan):
        spectrum = y [g]
        absFFT = np.abs(np.fft.fft(spectrum, n=2**zepopadding))
        if go == 'gauss':
            absFFT = gaussian_filter(np.abs(np.fft.fft(spectrum, n=Zero)),sigma = 18)
        #data_bscan.append(absFFT[int(len(absFFT)/1.25):len(absFFT)-cut_DC])
        data_bscan.append(absFFT)
        procent_old = round((g-1)*100/len(y))
        procent_new = round((g)*100/len(y))
    
        if procent_new > procent_old :
            print ('Please wait ' + str(100 - procent_new) + '% left')
            if procent_new == 94:
                print ('Almost there')
    return np.rot90(data_bscan,1)

Low_wavelength_cut = 780
High_wavelength_cut = 920

central_wavelength1 = 810
central_wavelength2 = 890

resolution =(2.5/(2**(p-11)))*10**(-6) # resolution in mm where 2.5 is a pix 


def gaussian(mu, sigma=0.8, makeit2048 = False):
    x = np.arange(Low_wavelength_cut, High_wavelength_cut, (High_wavelength_cut-Low_wavelength_cut)/len(dispCompVect))
    
    if makeit2048 == True: 
        x = np.arange(Low_wavelength_cut, High_wavelength_cut, (High_wavelength_cut-Low_wavelength_cut)/2048)
    
    return 1/(sigma * np.sqrt(2 * np.pi))*np.exp( - (x - mu)**2 / (2 * sigma**2) )


def dispersion(spectrum, ref_ind=1.508):
    x = np.arange(-5, 5, 10/len(spectrum)) #gaussian range

    mu1 = -5+(central_wavelength1-Low_wavelength_cut)*10/(High_wavelength_cut-Low_wavelength_cut)  # 
    mu2 = -5+(central_wavelength2-Low_wavelength_cut)*10/((High_wavelength_cut-Low_wavelength_cut))  
    
    sigma = 0.8
    
    gaussian1 = 1/(sigma * np.sqrt(2 * np.pi))*np.exp( - (x - mu1)**2 / (2 * sigma**2) )
    gaussian2 = 1/(sigma * np.sqrt(2 * np.pi))*np.exp( - (x - mu2)**2 / (2 * sigma**2) )
    ###### Algorithm starts
    
    
    spectrum1 = gaussian1*spectrum
    spectrum2 = gaussian2*spectrum
    
    absFFT1 = np.flip(np.abs(np.fft.fft(spectrum1, n=Zero)))
    absFFT2 = np.flip(np.abs(np.fft.fft(spectrum2, n=Zero)))
    

    cut = int(len(absFFT1)/2)

    index1 = peakutils.indexes(absFFT1[cut_DC:cut], thres=0.2, min_dist=2*(2**(p-10))) 
    index2 = peakutils.indexes(absFFT2[cut_DC:cut], thres=0.2, min_dist=2*(2**(p-10))) 
    
    #print ('z1 is ' + str(index1) + '\n z2 is ' + str(index2))
    c = scipy.constants.c # in m/s 

    omega1 = 2*np.pi*c/(central_wavelength1)  # wavelength in nm  
    omega2 = 2*np.pi*c/(central_wavelength2)  # wavelength in nm
    l_s = np.abs(index1[0]-index1[-1])* resolution*ref_ind  # in mm
    z_1 = (index1[-1])*resolution #in m
    z_2 = (index2[-1])*resolution
    walk_off = np.abs(z_2-z_1)*ref_ind
    
    beta_2 =(walk_off/(c*l_s*(omega1-omega2)))*10**(24)

    return walk_off 
walks= []
def walk_off(spectrum, where_to_look = 0,  ref_ind=1.508 ):
    x = np.arange(-5, 5, 10/len(spectrum)) #gaussian range

    mu1 = -5+(central_wavelength1-Low_wavelength_cut)*10/(High_wavelength_cut-Low_wavelength_cut)  # 
    mu2 = -5+(central_wavelength2-Low_wavelength_cut)*10/((High_wavelength_cut-Low_wavelength_cut))  
    
    sigma = 0.8
    
    gaussian1 = 1/(sigma * np.sqrt(2 * np.pi))*np.exp( - (x - mu1)**2 / (2 * sigma**2) )
    gaussian2 = 1/(sigma * np.sqrt(2 * np.pi))*np.exp( - (x - mu2)**2 / (2 * sigma**2) )
    ###### Algorithm starts
    
    
    spectrum1 = gaussian1*spectrum
    spectrum2 = gaussian2*spectrum
    
    absFFT1 = np.flip(np.abs(np.fft.fft(spectrum1, n=Zero)))
    absFFT2 = np.flip(np.abs(np.fft.fft(spectrum2, n=Zero)))
    
    roi_from = int(where_to_look-0.02*len(absFFT1))
    roi_to = int(where_to_look+0.02*len(absFFT1))
    walk_off = 0
    peak1= roi_from + np.argmax (absFFT1[roi_from:roi_to])

    peak2 = roi_from +np.argmax (absFFT2[roi_from:roi_to])


    if peak2 < peak1+200:
        plt.figure()
        plt.title('This Ascan has not been taken ' +' \n Select the peaks manually if you want to add')
        plt.plot(absFFT1)
        plt.plot(absFFT2)
    else:

        plt.figure()
        plt.title('Seems good' )
        plt.plot(absFFT1)
        plt.plot(absFFT2)

        walk_off = peak2-peak1

    return walk_off

def dispersion_new(walk_off, distance, resolution =(2.5/(2**(p-11)))*10**(-6), ref_ind=1.508):
    c = scipy.constants.c # in m/s 

    omega1 = 2*np.pi*c/(central_wavelength1*10**(-9))  # wavelength in m  
    omega2 = 2*np.pi*c/(central_wavelength2*10**(-9))  # wavelength in m
    l_s = (np.abs(distance)/ref_ind)
    print ('distance ' + str(l_s) + ' pixels')
    #l_s = 975  # in mm
    
    beta_2 =(walk_off/(c*l_s*(omega1-omega2)))
    return beta_2*10**(27)
    



def dispersionplus(spectrum, ref_ind=1.508):
    x = np.arange(-5, 5, 10/len(spectrum)) #gaussian range

    mu1 = -5+(central_wavelength1-Low_wavelength_cut)*10/(High_wavelength_cut-Low_wavelength_cut)  # 
    mu2 = -5+(central_wavelength2-Low_wavelength_cut)*10/((High_wavelength_cut-Low_wavelength_cut))  
    
    sigma = 0.8
    
    gaussian1 = 1/(sigma * np.sqrt(2 * np.pi))*np.exp( - (x - mu1)**2 / (2 * sigma**2) )
    gaussian2 = 1/(sigma * np.sqrt(2 * np.pi))*np.exp( - (x - mu2)**2 / (2 * sigma**2) )
    ###### Algorithm starts
    
    
    spectrum1 = gaussian1*spectrum
    spectrum2 = gaussian2*spectrum
    

    f1 = interp1d(phase_nonlin, spectrum1, fill_value='extrapolate')
    f2 = interp1d(phase_nonlin, spectrum2, fill_value='extrapolate')

    spectrum_lin1 = f1(phase_lin)
    spectrum_lin2 = f2(phase_lin)

# compensate dispersion

    #spectrum_lin_disp1 =spectrum_lin1*np.exp(-1j*dispCompVect)
    #spectrum_lin_disp2 =spectrum_lin2*np.exp(-1j*dispCompVect)    
    
    absFFT1 = np.flip(np.abs(np.fft.fft(spectrum_lin1, n=Zero)))
    absFFT2 = np.flip(np.abs(np.fft.fft(spectrum_lin2, n=Zero)))
    plt.figure()

    plt.plot(spectrum, c = 'r', linewidth=0.5)    
   
    cut_DC =  (2**(p-10))*21
    cut = int(len(absFFT1)/2)

    index1 = peakutils.indexes(absFFT1[cut_DC:cut], thres=0.9, min_dist=2*(2**(p-10)))  
    index2 = peakutils.indexes(absFFT2[cut_DC:cut], thres=0.9, min_dist=2*(2**(p-10))) 
    
    index_first = peakutils.indexes(absFFT1[cut_DC:int(cut-cut/2)], thres=0.9, min_dist=2*(2**(p-10))) 
    index_last = peakutils.indexes(absFFT1[int(cut-cut/2):cut], thres=0.9, min_dist=2*(2**(p-10))) 

    #print ('z1 is ' + str(index1) + '\n z2 is ' + str(index2))
    c = scipy.constants.c # in m/s 

    omega1 = 2*np.pi*c/(central_wavelength1)  # wavelength in nm  
    omega2 = 2*np.pi*c/(central_wavelength2)  # wavelength in nm
    l_s = (np.abs(index_first-index_last)*resolution)
    print ('distance is ' + str(l_s))
    #l_s = 975  # in mm
    z_1 = (index1[-1])*resolution #in m
    z_2 = (index2[-1])*resolution
    walk_off = np.abs(z_2-z_1)*ref_ind
    
    beta_2 =(walk_off/(c*l_s*(omega1-omega2)))*10**(24)

    return beta_2

def showme (data, title='title is missing', full = True, contrast = 5000):
    
    if isinstance(data[0], (np.ndarray)) or isinstance(data[0], list):
        plt.figure()
        if full == True:
            plt.imshow(data, cmap='gray')
        else:
            plt.imshow(data[:int(len(data)/3)], cmap='gray')
        plt.colorbar()
        plt.clim([0, contrast])
        plt.title(title)
        plt.axis('tight')
    

    elif isinstance(data[0], (int, float)):
        plt.figure()
        plt.title(title)
        cut = int(len(data)/2)
        if full == False:
            plt.plot(data[cut_DC:cut], c = 'r', linewidth=0.5)
        else:
            plt.plot(data,'-', c = 'r', linewidth=0.5) 
        plt.axis('tight')
        


def showmegraph(ascan_data, title='title is missing', full = True):

    plt.figure()
    plt.title(title)
    cut_DC =  (2**(p-10))*21
    cut = int(len(ascan_data)/2)
    if full == False:
        plt.plot(ascan_data[cut_DC:cut], c = 'r', linewidth=0.5)
    else:
        plt.plot(ascan_data, c = 'r', linewidth=0.5) 
    plt.axis('tight')


def showmegraphplus(ascan_data, title='title is missing', full = True):

    plt.figure()
    plt.title(title)
    cut_DC =  (2**(p-10))*21
    cut = int(len(ascan_data)/2)
    if full == False:
        fig, ax = plt.subplots()
        ax.plot(ascan_data[cut_DC:cut], c = 'r', linewidth=0.5)
        axins = zoomed_inset_axes(ax, 1.5, loc=2) # zoom-factor: 2.5, location: upper-left
        axins.plot(ascan_data[cut_DC:cut])
        x1, x2, y1, y2 = 120, 160, 4, 6 # specify the limits
        axins.set_xlim(x1, x2) # apply the x-limits
        axins.set_ylim(y1, y2) # apply the y-limits
        plt.yticks(visible=False)
        plt.xticks(visible=False)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        
        
        
    else:
        plt.plot(ascan_data, c = 'r', linewidth=0.5) 
    plt.axis('tight')
        


def showmeimage(bscan_data,  title='title for B-scan is missing', full = True):

    plt.figure()
    if full == True:
        plt.imshow(bscan_data, cmap='gray')
    else:
        plt.imshow(bscan_data[:int(len(bscan_data)/3)], cmap='gray')
    plt.colorbar()
    plt.clim([0, 18000])
    plt.title(title)
    plt.axis('tight')


def showmehist (data_hist, title_hist = 'Title is missing', numb_bins = None):
    plt.figure()
    plt.hist(data_hist, bins=numb_bins,  alpha=0.5, histtype='bar', ec='black')
    st_dev = np.std(data_hist)
    ave = np.mean(data_hist, dtype=np.float32)
    plt.title(str(title_hist) + '\n Standart deviation: %.3f \n Average: %.4f ' %(st_dev, ave))
    
    plt.show()
    
def showme3D (cscan, title_hist = 'Title is missing'):
    plt.hist(cscan, alpha=0.5, histtype='bar', ec='black')
    st_dev = np.std(cscan)
    ave = np.mean(cscan, dtype=np.float32)
    plt.title(str(title_hist) + '\n Standart deviation: %.2f \n Average: %.2f [$fs^2/mm$] ' %(st_dev, ave))
    
    plt.show()
    
    
def segmentation (image):

    img = cv2.imread(image)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    

    sure_bg = cv2.dilate(opening,kernel,iterations=500)
    

    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.05*dist_transform.max(),255,0)
    

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    

    ret, markers = cv2.connectedComponents(sure_fg)
    

    markers = markers+1
    
    markers[unknown==255] = 0
    
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    
    plt.imshow(img)
    
def first_peak (ascan, multi = 6):
    initial = ascan[0]*multi
    for i in range (0, len(ascan)):
        value = ascan[i]        
        if initial < value:
            break
    return i
            

def bscan_fast(y, first_ascan = 0, last_ascan = number_ascans, go = 0):

    data_bscan = []
    cut_DC =  (2**(p-10))*21
    
    for g in range (first_ascan, last_ascan):
        spectrum = y [g]
        absFFT = np.abs(np.fft.fft(spectrum, n=Zero))
        if go == 'gauss':
            absFFT = gaussian_filter(np.abs(np.fft.fft(spectrum, n=Zero)),sigma = 18)
        #data_bscan.append(absFFT[int(len(absFFT)/1.25):len(absFFT)-cut_DC])
        data_bscan.append(absFFT)
        procent_old = round((g-1)*100/len(y))
        procent_new = round((g)*100/len(y))
    
        if procent_new > procent_old :
            print ('Please wait ' + str(100 - procent_new) + '% left')
            if procent_new == 94:
                print ('Almost there')
                
    
    return np.rot90(data_bscan,1) 
   
sigma = 8
    
def walkoff (data, start_peak_range = 0, end_peak_range = 200):
    for i in range (start_peak_range, end_peak_range): 
        spectrum1 = gaussian(820, sigma)*calibrate[i]
        spectrum2 = gaussian(880, sigma)*calibrate[i]
        
        absFFT = fourrier(calibrate[i])
        absFFT1 = fourrier(spectrum1)
        absFFT2 = fourrier(spectrum2)
   
def omega(wave):
    
    c = 2.998*10**8 # m/s
    
    f = c/(wave*10**(-9))
    
    omega = 2*3.141592*f
    
    return omega

    
def Kalman(data, start, end):
    
   
    new_set = []
    E_measured = 2
    Estimeted_value = data[0][start]
    Estimated_error = 2
    
    for j in range (start, end):
        KG = Estimated_error/(Estimated_error+E_measured)
        Estimeted_value = Estimeted_value + KG*(data[0][j]-Estimeted_value)
        Estimated_error = (1-KG)*(Estimated_error)
        new_set.append(Estimeted_value)
    
   
    return new_set
    
    
def peak (row_data):
    
    point = np.argmax(row_data)
    
    return point


def peak_threshold(row_data , threshold, start = 0,  end = 1048):
    
    h = start
    
    value = row_data [h]

    while h < end and value < threshold:
        h+=1
        value = row_data[h]
      
    peak = h
  
    return peak

def are_neighbors_okay (row, confidence = 1):
    
    result = [True]
    for i in range (1, len(row)-1):
            
        if row [i]>(row [i-1] + confidence) and row [i]>(row [i+1] +confidence):
            res = False
        elif (row [i]+confidence)<(row [i-1]) and (row [i]+confidence)<row [i+1]:
            res = False
        else:
            res = True
        result.append(res)
    result.append(True)
    
    return result
        
    
def total_clean (data, limit1, limit2):
    
    data [0:limit1] = 0
    data [limit2:len(data)]= 0 
    return data

def improve_clean (data, limit1, limit2, factor = 4):
    
    for y in range (0, limit1):
        data[y] = data[y]/factor
    for y in range (limit2, len(data)):
        data[y] = data[y]/factor
        
    return data
    
    
def get_only_flat(refr, get_what):
    

    for i in range (0, len(b)):
        
        g = b[i]
        
        if g.startswith(("File: g")):
            
            r = (refr[i])
            
            r = r.replace("[]", "a")
            
            r = r.replace("[", "")
            
            r = r.replace("]", "") 
            
            list_ = r.split (",")
            
          
            all_refr.append((list_))
    
    flat_list = []
    for sublist in all_refr:
        for item in sublist:
            flat_list.append((item))  
            
    fd = 'a'
    
    flat_list.remove('a')
    
    
    new = []
    for i in range (0, len(flat_list)):
        
        if flat_list[i] != 'a':
            new.append(float(flat_list[i]))    
    

def load_folder(path):    
    
    
    all_files_path = str(path)+ '/*.txt'
    
    files = glob.glob(all_files_path)
    
    return files
    
data = [19]    
def sum_of_components_after(data, i=0, length= len(data)):
    if len(data) != length:
        sum_of_next = 0
        length = len(data)
        for j in range (i, length):
            sum_of_next+=data[j]
    else:
        sum_of_next = data [i]
    return sum_of_next
 
def absorb_coef(data, resolution = 1, start = 0, end = len(data)):
    mu = []
    
    end = len(data)
    data = data [start:end]
    if len(data) != end:
        for y in range (start, end-1):
            end = len(data)
            sum_of_next = 0.000000000001
            for j in range (y+1, end):
                sum_of_next+=data[j]
            coef = data[y]/(2*resolution*sum_of_next)
            mu.append(coef)
            
        
    else:
        for y in range (start, end-1):
            end = len(data)
            sum_of_next = 0.000000000001
            for j in range (y+1, end):
                sum_of_next+=data[j]
            
            coef = data[y]/(2*resolution*sum_of_next)
            mu.append(coef)
            
    
    return mu

data = []        
def fwhm_line(data, slice_from = 0, slice_to = len(data), thr = 0.95):
    
    i=0
    check = data [slice_from:slice_to]
    value = 0
   
    while value < thr*max(check):
        i+=1
        value = check[i]
    s = slice_from+i
    while value > thr*max(check):
        i+=1
        value = check[i]
    e = slice_from+i
    
    line = np.linspace(s, e, e-s)
    lenth = e- s 
    yline = [thr*max(check)]*len(line)
    return line, yline, e, s
            
    
    
    
    
         