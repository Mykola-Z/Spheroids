# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 00:00:03 2020

@author: mzly903
"""


# Short cuts for many things in OCT data processing
import OCTFunctions as octf

import numpy as np
# stable version 1.17.4 

from matplotlib import pyplot as plt
# stable version 3.1.1 

import scipy.stats as stats # gaussian

import csv # work with csv files to save data
# stable version 1.0


import glob

import datetime

time = str(datetime.datetime.now())
date_time = time[5:19].replace(":", "-")

class DataProcessing:

    """ All data preprocessing features """


    def __init__(self, data, data_name):

        self.data = data
        self.data_name = data_name


    def fourrier(self, Zero = 2048, first_ascan = 0, last_ascan = None, absolute = True):
        """" 
        if absolute == True: Returns a depth profile; 
        otherwise just Fourrier transform 
        """
        if last_ascan == None:
            last_ascan = len(self.data)

        if isinstance(self.data[0], int) or isinstance(self.data[0], float):
            depth_profile = np.fft.fft(self.data, n = Zero)
            if absolute:
                depth_profile = abs(depth_profile)

        else:
            depth_profile = []

            for i in range(first_ascan, last_ascan):
                fourrier_raw = np.fft.fft(self.data[i], n=Zero)
                if absolute:
                    fourrier_raw = abs(fourrier_raw)
                depth_profile.append(fourrier_raw)

        return depth_profile


    def gauss_window(self, cw, sigma, pixels = 2048, wavelength_from = 766.8, wavelength__to = 920):
        
        """" 
        Returns a list with Gaussian distribution
        
        """
        
        mu = cw
        x = np.linspace(wavelength_from, wavelength__to, pixels)
        gaussian_distribution = stats.norm.pdf(x, mu, sigma)
        max_ = max(gaussian_distribution)
        for l in range(len(gaussian_distribution)):
            gaussian_distribution[l] = gaussian_distribution[l]/max_

        return gaussian_distribution



    def apply_gauss_window(self, cw, sigma, pixels = 2048, wavelength_from = 766.8, wavelength__to = 920 ):
        """" 
        Returns a spectrum multiplied by a Gaussian window
        
        """
        if isinstance(self.data[0], int) or isinstance(self.data[0], float): 
            after_gaussian_spectrum = self.data*self.gauss_window(cw, sigma, pixels, wavelength_from, wavelength__to)

        else:
            after_gaussian_spectrum = []
            for i in range(len(self.data)):
                new_ascan = self.data[i]*self.gauss_window(cw, sigma, pixels, wavelength_from)
                after_gaussian_spectrum.append(new_ascan)

        return after_gaussian_spectrum
    
    
""" Helps the programme to understand where is the object    
Includes methods with semi-automatic visualisation        
"""    
class ComputerVision:
   
    def __init__(self, fourrier_data, data_name):
        
        self.fourrier_data = fourrier_data
        self.name = data_name
        
    def geometrical_line(self):
        
        bscan = self.fourrier_data[:]
        
        l_values = []
        
        for ii in range(0,4):
            crop_raw_left = bscan[ii][int(len(bscan[0])/20):int(len(bscan[0])/3)]
            value = np.argmax(crop_raw_left) + int(len(bscan[0])/20)
            l_values.append(value)
        r_values = []
        for jj in range(len(bscan)-4, len(bscan)):
            crop_raw_right = bscan[jj][int(len(bscan[0])/20):int(len(bscan[0])/3)]
            value = np.argmax(crop_raw_right)  + int(len(bscan[0])/20)
            r_values.append(value)
        
        l_values.remove(max(l_values))
        l_values.remove(min(l_values))
        
        r_values.remove(max(r_values))
        r_values.remove(min(r_values))
        
        left_point = int(sum(l_values)/len(l_values))
        right_point = int(sum(r_values)/len(r_values))

        line = [left_point] 

        line_sp = np.linspace(left_point, right_point, len(bscan)-2)
        line.extend(line_sp)
        line.append(right_point)
        
        return line


    def threshlold_segmentation(self, thr_value = 0.67, smoothing = True):
        """"
        Algoritm cuts first 5% of the depth profile(to get rid of noise peak)
        Algoritm cuts second half of the depth profile(to get rid of mirror image)
            UPDATES: CUTS last 2/3, spheroid is located on the first 1/3 of the image 
    
        
        Algoritm sets a threshold value (thr_value*100)% of the highest peak in an Ascan
        
        Algoritm goes from the beginning of croped image (or Ascan) to 
        the point threshold value
        
        Algoritm skips noise points before the surface:
            It counts if "above threshold" point has been met skip_pixels times in succession.
            where skip_pixels = 0.045% of the length of initial data raw (e.g for 4096 pixels depth-profile = 2 pixels, for 65536 = 30)
        
        WHEN "above threshhold" value has been met skip_pixels times it returns coordinate of the first "correct" surface
        
        In the same way bottom surface is detected. The cropped raw is turned around
        
        
        """

        abscan = self.fourrier_data[:]


        if isinstance(abscan[0], int) or isinstance(abscan[0], float):
            skip_pixels = int(len(abscan)*0.0005 + 1) # (for 4096 pix: skips 2 pixels, for 65536 skips 33 pixels )
            
            for_segmentation = abscan[:int(len(abscan)/3)] # crop mirror effect
            for_segmentation = for_segmentation[int(len(for_segmentation)/30):] # crop peak at the beggining
            threshold = thr_value*max(for_segmentation)
            i = 0
            check_value = 0
            confidence_interval = 0
            while confidence_interval < skip_pixels and i < len(for_segmentation):
                check_value = for_segmentation[i]
                
                if check_value > threshold:
                    confidence_interval += 1
                else:
                    confidence_interval = 0 # if the next point is below threshold
                i += 1
            top_surface = int(len(for_segmentation)/30) + i - skip_pixels

            ############## Bottom surface #################

            for_b_segmentation = for_segmentation[:(i-skip_pixels):-1] # reverse data raw and slice untill the top surface peak
            
            threshold = thr_value*max(for_b_segmentation)
            i = 0
            check_value = 0
            confidence_interval = 0
            while confidence_interval < skip_pixels and i < len(for_b_segmentation):
                check_value = for_b_segmentation[i]
    
                if check_value > threshold:
                    confidence_interval += 1
                else:
                    confidence_interval = 0 # if the next point is below threshold
                i += 1
                
            bottom_surface = int(len(abscan)/3) - i + skip_pixels

        else:
            
            """
            
            Define the line of petry dish 
            
            Top surface is always above this line
            
            """

            line = self.geometrical_line()
            
            """ image surface detection"""
            
            top_surface = []
            bottom_surface = []
            skip_pixels = int(len(abscan[0])*0.0007 + 1) # (for 4096 pix: skips 2 pixels, for 65536 skips 33 pixels )

            for y in range(len(abscan)):

                for_segmentation = abscan[y][int(len(abscan[0])/30):int(line[y])] # consider points only untill the petri dish
                threshold = thr_value*max(for_segmentation)
                i = 0
                check_value = 0
                confidence_interval = 0
                
                
                while confidence_interval < skip_pixels  and i < len(for_segmentation):
                    check_value = for_segmentation[i]
                    
                    if check_value > threshold:
                        confidence_interval += 1 # to ignore noise points before the surface 
                    # else:
                    #    confidence_interval = 0 # if the next point is below threshold
                    i += 1
                detected_point_one_raw = int(len(abscan[0])/30) + i - skip_pixels
                top_surface.append(detected_point_one_raw)
                
                ################## Bottom surface ########### 
                
                for_b_segmentation = abscan[y][int(line[y]):int(len(abscan[y])/3)] # reverse data raw and slice untill the top surface peak
                for_b_segmentation = for_b_segmentation[::-1]
                
                thr_value = 0.66
                threshold = thr_value*max(for_b_segmentation)
                i = 0
                check_value = 0
                confidence_interval = 0
                while confidence_interval < (skip_pixels-10) and i < len(for_b_segmentation):
                    check_value = for_b_segmentation[i]
        
                    if check_value > threshold:
                        confidence_interval += 1
                    else:
                        confidence_interval = 0 # if the next point is below threshold
                    i += 1
                    
                detected_point_b_surface = int(len(abscan[0])/3) - i + skip_pixels
                bottom_surface.append(detected_point_b_surface)
                
                
        ############### Smoothing ############
        if smoothing:
            
            confidence =  top_surface[0]*0.02      
            smooth_top = octf.are_neighbors_okay(top_surface, confidence)
            for l in range(len(smooth_top)):
                if smooth_top[l] == False:
                    top_surface[l] = (top_surface[l-1]+top_surface[l+1])/2
            smooth_bottom =  octf.are_neighbors_okay(bottom_surface, confidence)
            for e in range(len(bottom_surface)):
                if smooth_bottom[e] == False:
                    bottom_surface[e] = (bottom_surface[e-1]+bottom_surface[e+1])/2      
                
        return (top_surface, bottom_surface, line) 



    def show(self, first_ascan = 0, last_ascan = None):
        """" 
        if absolute == True: Returns depth profile; 
        otherwise just Fourrier transform 
        """
        if last_ascan == None:
            last_ascan = len(self.fourrier_data)

        if isinstance(self.fourrier_data[0], int) or isinstance(self.fourrier_data[0], float):
            plt.figure()
            plt.plt(self.fourrier_data)
            plt.title(self.name)
        else:
            plt.figure()
            plt.imshow(np.rot90(self.fourrier_data), cmap="gray")
            plt.clim([0, 10000])
            plt.axis("tight")
            plt.title(self.name)
    
    def where_to_crop(self):
        
        """ 
        Returns 2 coordinates 
        
        Left-upper and right-lower ((x1, x2, y1, y2))
        
        of the cut-window
        
        """
        

        self.show()
        crop_window = plt.ginput(2)
        x1 = int(crop_window[0][0])
        x2 = int(crop_window[1][0])
        y1 = int(crop_window[0][1])
        y2 = int(crop_window[1][1])

        return (x1, x2, y1, y2)

    def crop_image(self):
        
        """ 
        Returns cropped image data raws (pixels)
        
        """
        l = self.where_to_crop()
        x_axis_cut = np.array(self.fourrier_data[l[0]:l[1]])

        # return x_axis_cut[:, l[2]:l[3]]
        return x_axis_cut[:]


class DataFeautures(ComputerVision):
    
    
    def __init__(self, data_raw, name):
        
        self.data_raw = data_raw
        self.name = name
    
    def build_line(point1, point2):
        line = np.linspace(int(point1), int(point2), int(point2[0]) - int(point1[0] - 1))
        return line
    
    def peak_detection(self, from_ = 0, to = None):
        if to == None:
            to = len(self.data_raw)
        peak = np.argmax(self.data_raw[from_:to])
        return peak
    

    def refractive_index(peak1, peak2, geometrical_point):
        return (peak2-peak1)/(geometrical_point-peak1)

def peak_detection(row, from_ = 0, to = None):
    if to == None:
        to = len(row)
    peak = from_ + np.argmax(row[from_:to])
    return peak        

def geom_line(bscan):

    l_values = []
    
    for ii in range(0,4):
        crop_row_left = bscan[ii][int(len(bscan[0])/20):int(len(bscan[0])/3)]
        value = np.argmax(crop_row_left) + int(len(bscan[0])/20)
        l_values.append(value)
    r_values = []
    for jj in range(len(bscan)-4, len(bscan)):
        crop_row_right = bscan[jj][int(len(bscan[0])/20):int(len(bscan[0])/3)]
        value = np.argmax(crop_row_right)  + int(len(bscan[0])/20)
        r_values.append(value)
    
    l_values.remove(max(l_values))
    l_values.remove(min(l_values))
    
    r_values.remove(max(r_values))
    r_values.remove(min(r_values))
    
    left_point = int(sum(l_values)/len(l_values))
    right_point = int(sum(r_values)/len(r_values))

    line = [left_point] 

    line_sp = np.linspace(left_point, right_point, len(bscan)-2)
    line.extend(line_sp)
    line.append(right_point)
    
    return line

def vertical_line(x, y):

    return [[x]*int(y), [y]*int(y)]

def refractive_ind(peak1, peak2, geometrical_point):
    return (peak2-peak1)/(geometrical_point-peak1)

def norm_inten(data_row):
    print("start")
    return [float(i)*40/max(data_row) for i in data_row]

def norm_gvd(data1, data2, peak1, peak2):
    dif = int((int(peak1) - int(peak2)))
    l = -abs(dif)
    if dif >= 0:
        mod_data1 = data1[:l]
        mod_data2 = data2[dif:]
    else:
        mod_data1 = data1[abs(dif):]
        mod_data2 = data2[:]
    return [mod_data1, mod_data2, dif]

### Load all d (day 7) spheroids ###





######################################
    #################################
#################################
    ####################
    # CHANGE DIRECTORY With the files##
#############################
    ############################
########################





files = glob.glob('C:/Users/mzly903/Desktop/PhD/2. Data/1. Spheroids/Experiment 4 21 Jan 2017/21JanMeasurements/d1*.txt')
# with open('spheroids' + date_time +'.csv', 'a', newline='') as newFile:
#     newFileWriter = csv.writer(newFile)
#     newFileWriter.writerow(['name', 'Ascan', "cw1", "cw2" ,"sigma" ,'ri',
#                             "Oprtical_distance1","Oprtical_distance2",'walks normalized',
#                             'Geometrical Distance', 'Optical Distnace', ''])
for yu in files[:1]:
# upload file and get its name
    
    file = np.loadtxt(yu)
    name = yu[-10:]

    zero_pad_image = 15  #### power degree of the zero padding values eg 2**15
    proccessed_data = DataProcessing(file, name)  #### create a DataProcessing object
    bscan = proccessed_data.fourrier(Zero = 2**zero_pad_image) #### apply method to get bscan
    recognition = ComputerVision(bscan, name + str('\n DO 2 clicks around TOP LEFT and Bottom Right of the spheroid \n and then enter Ascan you want to work on')) #### create a ComputerVision object
   
    b = np.array(recognition.where_to_crop()) ## recognition part
    line_geo = geom_line(bscan[b[0]: b[1]]) ## geometrical line for RI
#make sure the image is ok   
#     plt.figure()

#     plt.imshow(np.rot90(bscan[b[0]:b[1]]), cmap = "gray")
#     plt.clim([0, 10000])
#     plt.axis("tight")
# # draw a line for geometrical thickness
#     plt.plot(line_geo)
#     plt.title("Make sure line is properly detected and click anywhere")
# # this ginput just to stop a program and see the image
#     ttt = plt.ginput()

    
    # print(ttt)
    # if ttt[0][0] < 3:
    #     print("Bscan repeat", yu)
    #     filename =  'dataf1.txt'   

    #     with open(filename, "w+") as json1:
    #         text = "Repeat " + str(yu)
    #         json1.write(text)

    #     continue
        
# declare new object of CombuterVision class
    croped_signal = ComputerVision(bscan[b[0]:b[1]], name)
# get surface interpolation points to show on the Ascan    
    geometrical_points = croped_signal.geometrical_line()
    
    print(name)
    #### Work with every Ascan to collect as much data as possible ####
    sigma1 = 30
    sigma2 = 15
    cw_l1 = 814
    cw_l2 = 830
    cw_h1 = 884
    cw_h2 = 870
    
    def g_window(cw, sigma, pixels = 2048, wavelength_from = 766.8, wavelength__to = 920):
        
        """" 
        Returns a list with Gaussian distribution
        
        """
        
        mu = cw
        x = np.linspace(wavelength_from, wavelength__to, pixels)
        gaussian_distribution = stats.norm.pdf(x, mu, sigma)
        max_ = max(gaussian_distribution)
        for l in range(len(gaussian_distribution)):
            gaussian_distribution[l] = gaussian_distribution[l]/max_

        return gaussian_distribution
    show_gauss1 = g_window(cw_l1, sigma1)
    show_gauss2 = g_window(cw_l2, sigma1)
    show_gauss3 = g_window(cw_h1, sigma1)
    show_gauss4 = g_window(cw_h2, sigma1)
    show_gauss5 = g_window(cw_l1, sigma2)
    show_gauss6 = g_window(cw_l2, sigma2)
    show_gauss7 = g_window(cw_h1, sigma2)
    show_gauss8 = g_window(cw_h2, sigma2)
  
        
    for_check = int(input("Enter central Ascan: "))
    check = file[for_check]
    x = np.linspace(766.8, 920, 2048)
    for yt in range(len(check)):
        check[yt] = check[yt]/max(check)
   #  plt.figure()
   #  plt.plot(x,check)
   #  plt.plot(x,show_gauss1, "b")
   #  #plt.plot(x,show_gauss2, "b")
   #  plt.plot(x,show_gauss3, "r")
   #  #plt.plot(x,show_gauss4, "r")
   #  #plt.plot(x,show_gauss5, "b")
   #  #plt.plot(x,show_gauss6, "b")
   # # plt.plot(x,show_gauss7, "r")
   # # plt.plot(x,show_gauss8, "r")
   #  plt.axvline(x=cw_l1, c = "b")
   #  #plt.axvline(x=cw_l2, c = "b")
   #  plt.axvline(x=cw_h1, c = "r")
    #plt.axvline(x=cw_h2, c = "r")
    # plt.title(f"Spectrum {for_check} and all Gaussian Windows")
    zero_pad_graph = 19
    specrum = DataProcessing(file[for_check], name)
    wave_low = specrum.apply_gauss_window(cw = 815, sigma=70)
    profile_low = DataProcessing(wave_low, name)
    graph_low = specrum.fourrier(Zero = 2**zero_pad_graph)
    plt.figure()
    plt.plot(graph_low[:int(len(graph_low)/2)])
    plt.title("4 Clicks to select front surface(between first 2 clicks ) and back surface")
    surfaces = plt.ginput(4)
    
    for ascan in range(for_check, for_check+1):
        
        cw_high = 880

        specrum = DataProcessing(file[ascan], name) # new object of DataProcessing class
        peaks1 = []
        peaks2 = []
        
        wave_high = specrum.apply_gauss_window(cw = cw_high, sigma=sigma1) #gaussian 1

        profile_high = DataProcessing(wave_high, name) # depth profile gaussain 1 # depth profile full spectrum

        # Increase zero padding 

        graph_high = profile_high.fourrier(Zero = 2**zero_pad_graph)

        for to_ave in range(500):
            step = 0.01 # in nm
            wave_range = 5  #in nm
            wave_high_add = specrum.apply_gauss_window(cw = cw_high-wave_range/2+to_ave, sigma=sigma1) #gaussian 1
            profile_high_add = DataProcessing(wave_high_add, name) # depth profile gaussain 1 # depth profile full spectrum
            graph_high_add = profile_high.fourrier(Zero = 2**zero_pad_graph)
            graph_high += graph_high_add
                    
  
        for iteration in range(1):
            
            cw_low = 820+5*iteration

            wave_low = specrum.apply_gauss_window(cw = cw_low , sigma=sigma1) #gaussian 1

            profile_low = DataProcessing(wave_low, name) # depth profile gaussain 1 # depth profile full spectrum

        # Increase zero padding 

            graph_low = profile_low.fourrier(Zero = 2**zero_pad_graph)
            
            
            for to_ave_low in range(500):
                step = 0.01 # in nm
                wave_range = 5  #in nm
                wave_low_add = specrum.apply_gauss_window(cw = cw_low-wave_range/2+to_ave_low, sigma=sigma1) #gaussian 1
                profile_low_add = DataProcessing(wave_low_add, name) # depth profile gaussain 1 # depth profile full spectrum
                graph_low_add = profile_low.fourrier(Zero = 2**zero_pad_graph)
                graph_low += graph_low_add
                    
            
        #     for to_aver in range (500):
                
        #         step = 0.01 #in nm
        #         wave_range = 5 # in nm
                
        #         wave_low_add = specrum.apply_gauss_window(cw = 795+5*iteration, sigma=sigma1) #gaussian 1

        #         profile_low = DataProcessing(wave_low, name) # depth profile gaussain 1 # depth profile full spectrum

        # # Increase zero padding 

        #         graph_low = profile_low.fourrier(Zero = 2**zero_pad_graph)
                                
    
            
            cut_low2 = graph_low[int(surfaces[2][0]):int(surfaces[3][0])]
            cut_high2 = graph_high[int(surfaces[2][0]):int(surfaces[3][0])]
            
            plt.figure()
            plt.suptitle(name + " " + "Ascan: " + str(ascan)+ 'Low cw: ' + str(cw_low) + "nm; High cw: "+ str(cw_high)+ "nm; sigma: " + str(sigma1))
            plt.subplot(221)
            x1 = np.arange(int(surfaces[0][0]), int(surfaces[1][0]), 1)
            plt.plot(x1, graph_low[int(surfaces[0][0]):int(surfaces[1][0])], c = "b")
            plt.plot(x1, graph_high[int(surfaces[0][0]):int(surfaces[1][0])], c = "r")

            plt.title('Front surfaces')
            cut_low1 = graph_low[int(surfaces[0][0]):int(surfaces[1][0])]
            cut_high1 = graph_high[int(surfaces[0][0]):int(surfaces[1][0])]
            plt.subplot(222)
            x2 = np.arange(int(surfaces[2][0]), int(surfaces[3][0]), 1)
            plt.plot(x2, graph_low[int(surfaces[2][0]):int(surfaces[3][0])], c ="b")
            plt.plot(x2, graph_high[int(surfaces[2][0]):int(surfaces[3][0])], c ="r")

            plt.title('Back surfaces')
            plt.subplot(223)
            correlation1 = np.correlate(cut_low1, cut_high1, 'same')
            correlation2 = np.correlate(cut_low2, cut_high2, 'same')
            peak1 = np.argmax(correlation1)
            if iteration == 0:
                peaky1 = peak1
                plt.text(x1[0], int(max(correlation1)/2) ,"Walk "+str(peak1-int(len(correlation1)/2)))
               
            else:
                plt.text(x1[0], int(max(correlation1)/2) ,"Walk "+str(peak1-int(len(correlation1)/2)))
                
            plt.plot(x1, correlation1)
            plt.axvline(x = x1[0]+peak1, c = "b")
            for iter_peaks1 in range(len(peaks1)):
                plt.axvline(x = x1[0]+ peaks1[iter_peaks1], c = "r")
            peaks1.append(peak1) 
            
            plt.subplot(224)
            peak2 =np.argmax(correlation2)
            if iteration == 0:
                peaky2 = peak2
                plt.text(x2[0], int(max(correlation2)/2) ,"Disp: "+str(peak2-int(len(correlation2)/2)))
            
            else:
                plt.text(x2[0], int(max(correlation2)/2) ,"Disp: "+str(peak2-int(len(correlation2)/2)))
            
            plt.plot(x2, correlation2)

            plt.axvline(x = x2[0]+ peak2, c = "b")

            for iter_peaks2 in range(len(peaks2)):
                plt.axvline(x = x2[0] + peaks2[iter_peaks2], c = "r")
            peaks2.append(peak2) 
            # plt.savefig("source_images\\"+str(cw_low)+"all.png", bbox_inches='tight')


        plt.show()

