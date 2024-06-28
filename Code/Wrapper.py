#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Miheer Diwan
MS Robotics Engineering,
WPI

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn.cluster 
 

# def read_img():
#     images  = []
#     for i in range(1,11):
#         path = "Images/"+ str(i) + ".jpg"
#         name = str(i)+'.jpg'
#         img = cv.imread(path)
#         images.append(img)   
#         # cv.imshow(name,img)
#         # cv.waitKey(0) 
#     return images

def ROT(ip,k_size,orientations=None): # Takes ip image, k size and orientations as input and outputs rotated images and pyplot in grayscale
    orientations = 1 if orientations is None else orientations

    bank = []
    # if orientations is None:
    #     cv.imshow(str(ip),ip)
    #     cv.waitKey(0)
    # else:    
    # plt.figure()
    for i in range(orientations):
        step = 360/orientations
        r,c = np.shape(ip)
        center = ((c-1)/2,(r-1)/2)
        M = cv.getRotationMatrix2D(center,(i*step),1.0)
        temp = cv.warpAffine(ip,M,(c,r))
        n_temp = cv.normalize(temp, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)
        n_temp= n_temp.astype(np.uint8) 
        bank.append(temp)
        # path = 'Code/DoG_Filter_Bank' 
        # name = str(n_temp) + str(k_size)+'_' + str(i+1) + '.jpg'
        # plt.subplot(2,orientations,i+1)       
        # plt.imshow(n_temp,cmap='gray',vmin = 0, vmax = 255)
    # plt.show()
    # cv.waitKey(0)
    return bank 

def gauss(k_size,sig_x=None,sig_y=None): # Returns Gaussian Kernel
    l = np.arange(-(k_size-1)/2,(k_size+1)/2,step=1,dtype=int)
    # print(l)
    sig_x = 1.0 if sig_x is None else sig_x
    sig_y = 1.0 if sig_y is None else sig_y
    g = []

    ##  verified with fspecial fn. in MATLAB
    for y in l:
        for x in l:    
            e = -0.5*((x**2/(sig_x**2))+(y**2/(sig_y**2))) 
            g.append((1/(2*np.pi*sig_x*sig_y))*np.exp(e))
    g = (np.array(g).reshape(k_size,k_size)) /np.sum(g)
    norm_g = cv.normalize(g, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)
    norm_g = norm_g.astype(np.uint8)
    # cv.imshow('Gaussian Kernel',norm_g) # can be commented out
    # cv.waitKey(0)
    return g

def DOG(ip):# Returns DOG
    ## Sobel Kernel:
    gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype='float')
    dog = cv.filter2D(ip,-1,gx) 
    # print(dog)

    n_dog = cv.normalize(dog, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)
    n_dog = n_dog.astype(np.uint8)
    # print(n_dog)
    # n_dog = cv.resize(n_dog,(2,21))
    # rot(n_dog,k_size,orientations)
    return dog
            
def LOG(k_size,sig): # return Laplacian of Gaussian
    
    lap_k = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    g = gauss(k_size,sig,sig)
    log = cv.filter2D(g,-1,lap_k)

    n_log = cv.normalize(log, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)
    n_log = n_log.astype(np.uint8)
    # cv.imshow('LoG',n_log)
    # cv.waitKey(0)
    return log 

def LMS(k_size,orientations): #returns LMS filter bank
    lms = []
    sigma = [1,np.sqrt(2),2,2*np.sqrt(2)]

    n_sig = []
    for i in sigma:
        n_sig.append(i*3)
    n_sig =  sigma + n_sig
    # print(n_sig)
    c = 1

    # # Making FDOG and SDOG with elongation factor of 3:
    for i in sigma[0:3]:
        # print('DOG: '+'sigma = ',i)
        g = gauss(k_size,i,3*i)
        dog = DOG(g) #fdog        
        dog_1 = ROT(dog,k_size,6)
        lms += dog_1
        sdog = DOG(dog)#sdog
        sdog_1 = ROT(sdog,k_size,6)
        lms += sdog_1
    
    # lms += tempCalculating gaussian kernel: 

    # # Making 8 Laplacian of Gaussians:
    # print('Making LoG')
    for i in n_sig:
        
        # print('scale =',i)
        log = LOG(k_size,i)
        log_1 = ROT(log,k_size,1) 
        lms += log_1

    # # Making 4 Gaussians:
    # print('Gaussians:')
    for i in sigma: 
        # print('sigma = ',i)        
        g = gauss(k_size,i,i)
        g_1 = ROT(g,k_size,1) 
        lms += g_1

    return lms

# lms = LMS(59,6)

def LML(k_size,orientations): # returns LML filter bank
    sigma = [np.sqrt(2),2,2*np.sqrt(2),4]

    lml = []

    n_sig = []
    for i in sigma:
        n_sig.append(i*3)
    n_sig =  sigma + n_sig
    # print(n_sig)
    c = 1

    # # Making FDOG and SDOG with elongation factor of 3:
    for i in sigma[0:3]:
        # print('DOG: '+'sigma = ',i)
        g = gauss(k_size,i,3*i)
        dog = DOG(g) #fdog        
        dog_1 = ROT(dog,k_size,6)
        lml += dog_1
        sdog = DOG(dog)#sdog
        sdog_1 = ROT(sdog,k_size,6)
        lml += sdog_1
    
    # # Making 8 Laplacian of Gaussians:
    # print(' Making LoG')
    for i in n_sig:
        # print('Sigma =',i)
        log = LOG(k_size,i)
        log_1 = ROT(log,k_size,1) 
        lml += log_1

    # # Making 4 Gaussians:
    # print(' Gaussians:')
    for i in sigma: 
        # print('Sigma = ',i)        
        g = gauss(k_size,i,i)
        g_1 = ROT(g,k_size,1) 
        lml += g_1

    return lml
# LML(89,6)

def GABOR(k_size,sig,theta,lamda,gamma,psi): #retursn Gabor Filters
    l = np.arange(-(k_size-1)/2,(k_size+1)/2,step=1,dtype=int)
    # g = gauss(k_size,1,1)
    gbor = []
    for y in l:
        for x in l:   
            x_ = x*np.cos(theta)+y*np.sin(theta) 
            # print('x_=',x_)
            y_ = -x*np.sin(theta) + y * np.cos(theta)
            # print('y_=',y_)
            e1 = -(x_**2 + gamma**2*y_**2)/(2*sig**2)
            # print('e1=',e1)
            e2 = (2*np.pi*x_/lamda) + psi
            # print('e2=',e2)
            gbor.append(np.exp(e1)*np.exp(1j*e2))
            # print('gbor=',gbor)
    gbor = (np.array(gbor,dtype=float).reshape(k_size,k_size))
    norm_gbor = cv.normalize(gbor, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)
    norm_gbor = norm_gbor.astype(np.uint8)     
    gbor =  ROT(norm_gbor,k_size,6)
    return gbor

def filter_bank():
    f_bank = []
    # # # ODOG Filter Bank:
    g = gauss(9,1,1)   
    dog = DOG(g)
    odog_1 = ROT(dog,k_size= 9,orientations = 16)
    # plt.imshow(odog_1)
    # plt.show()

    g = gauss(15,1,1)   
    dog = DOG(g)
    odog_2 = ROT(dog,k_size= 15,orientations = 16)

    odog = odog_1 + odog_2
    # plt.imshow(odog)
    # plt.show()

    # # # LM Filter Bank:
    lms = LMS(59,6)
    lml = LML(89,6)
    # print(len(lms))
    # print(len(lml))

    # # # Gabor Filter:
    gb1 = GABOR(k_size = 49,sig = 7,theta = 0,lamda = np.pi/4,gamma = 0.8,psi = 0)
    gb2 = GABOR(k_size = 49,sig = 7,theta = 0,lamda = np.pi/6,gamma = 0.8,psi = 0)
    gb3 = GABOR(k_size = 49,sig = 5,theta = 0,lamda = np.pi/4,gamma = 0.8,psi = 0)
    gb4 = GABOR(k_size = 49,sig = 5,theta = 0,lamda = np.pi/6,gamma = 0.8,psi = 0)

    gb5 = GABOR(k_size = 63,sig = 7,theta = 0,lamda = np.pi/4,gamma = 0.8,psi = 0)
    gb6 = GABOR(k_size = 63,sig = 7,theta = 0,lamda = np.pi/6,gamma = 0.8,psi = 0)
    gb7 = GABOR(k_size = 63,sig = 5,theta = 0,lamda = np.pi/4,gamma = 0.8,psi = 0)
    gb8 = GABOR(k_size = 63,sig = 5,theta = 0,lamda = np.pi/6,gamma = 0.8,psi = 0)

    gb = gb1 + gb2 + gb3 + gb4 + gb5 + gb6 + gb7 + gb8
    f_bank = odog_1 + odog_2 + lms + lml + gb1 + gb2 + gb3 + gb4 + gb5 + gb6 + gb7 + gb8
    
    # plt.show()
    # print('Filter Bank Size =',len(f_bank))
    return f_bank,odog,lms,lml,gb

def Make_Half_Disk(i):

    h_disk = []
    disk = np.zeros((i,i))
    x,y = np.shape(disk)
    x_mid = int((x-1)/2)
    y_mid = int((y-1)/2)
    center = (x_mid,y_mid)   
    # print(center)
    disk = cv.circle(disk,center, int((i-1)/2), (1,1,1), thickness=-1)
    # disk = disk * 
    disk[0:y,0:x_mid] = 0
    # disk = cv.normalize(disk, None, alpha = 0, beta = 1, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)

    disk_flip = cv.flip(disk,-1)
    # disk = cv.normalize(disk, None, alpha = 0, beta = 1, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)


    disk_1 = ROT(disk,i,8)
    # print(np.shape((disk_1))) 
    disk_2 = ROT(disk_flip,i,8)

    for j in range(len(disk_1)):
        # h_disk = disk_1[j] + disk_2[j]
        h_disk.append(disk_1[j])
        h_disk.append(disk_2[j])
    # print(np.shape(h_disk))
    return h_disk

def Half_Disk_Bank():

    hd1 = Make_Half_Disk(11)
    hd2 = Make_Half_Disk(17)
    hd3 = Make_Half_Disk(29) 

    
    half_disk_bank = hd1 + hd2 + hd3
    # for i in half_disk_bank:
    #     cv.imshow('h_disk',i)
    #     print('h_disk shape',np.shape(i))

        # cv.waitKey(0)
    return(half_disk_bank)

# # # Making Brightness Map
def Brightness_Map(img):
    b_map = []
    # images = read_img()
    # for img in images: 
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    x,y = np.shape(gray)
    ip = gray.reshape([x*y,1])
    # plt.figure()
    labels = sklearn.cluster.KMeans(n_clusters = 16, init = 'random', n_init = 'auto').fit_predict(ip)
    # labels = kmeans.predict(ip)
    b_img = labels.reshape([x,y])
    b_map.append(b_img)
    plt.imsave('Results/B_Map/BMap'+name+'.png',b_img)

    # plt.imshow(b_img)
    # plt.title('Brightness Map')
    # plt.show()
    return b_map

def Color_Map(img):
    # print('Making Colour Map')
    c_map = []
    # images = read_img()
    # for img in images: 
    x,y,c = np.shape(img)
    ip = img.reshape([x*y,c])
    # plt.figure()     
    labels = sklearn.cluster.KMeans(n_clusters = 16, init = 'random', n_init = 'auto').fit_predict(ip)
    # labels = kmeans.predict(img)
    c_img = labels.reshape([x,y])
    c_map.append(c_img)
    plt.imsave('Results/C_Map/CMap_'+name+'.png',c_img)

    # plt.imshow(colour_img)
    # plt.title('Color Map')
    # plt.show()
    return c_map 

def Texton_Map(img):
    # images = read_img()
    t_map = []
    f_bank,odog,lms,lml,gb = filter_bank()

    # for img in images: 
    # img = cv.imread('Images/7.jpg')
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # cv.imshow('gray',gray)
    # cv.waitKey(0)

    f_out = []
    # print('Filtering..')
    for f in f_bank:
        # print('Filtering..')
        f_img = cv.filter2D(gray,-1,f)
        f_out.append(f_img)
        # cv.waitKey(0)
    f_out = np.array(f_out)
    # print(np.shape(f_out))

    # plt.figure()
    # print('clustering')
    f,x,y = np.shape(f_out) 
    # print('f_out shape = ',np.shape(f_out))
    ip = f_out.reshape([f,x*y])
    # print('ip shape = ',np.shape(ip))
    ip = ip.T
    # print('ip.T shape = ',np.shape(ip))
    labels = sklearn.cluster.KMeans(n_clusters = 64, init = 'random', n_init = 2).fit_predict(ip)
    t_img = labels.reshape([x,y])
    t_map.append(t_img)
    plt.imsave('Results/T_Map/TMap_'+ name +'.png',t_img)
    # plt.imshow(t_img)
    # plt.title('Texton Map')
    # plt.show()
    return t_map

def Chi_Square(f_bank,bins,ip_img):
    chi_sq_dist = []
    offset = np.ones(np.shape(ip_img))*np.exp(-7)
    bin_off = np.min(ip_img)
    # print(bin_off)
    # chi_sq = 0*ip_img
    # cv.imshow('inp',ip_img)
    # cv.waitKey(0)
    for i in range(len(f_bank)-1):
        j = i + 1
        chi_sq = 0.0*ip_img

        l_mask = f_bank[i]
        r_mask = f_bank[j]
        
        for bin in range(bins):
            tmp = np.zeros(np.shape(ip_img))
            tmp[ip_img == bin+bin_off] = 1.0
            g = cv.filter2D(tmp,-1,l_mask)
            h = cv.filter2D(tmp,-1,r_mask)
            chi_sq += ((g-h)**2/(g+h+offset))/2
        
        chi_sq_dist.append(chi_sq)

        # cv.imshow('tmp',np.mean(chi_sq_dist,axis = 0))
        # cv.waitKey(0)
        i += 1
        # print(chi_sq_dist)
    return chi_sq_dist

hd_bank =  Half_Disk_Bank()
images = range(10)

for i in images:
    name = str(i+1)
    canny_name = 'CannyBaseline/'+ str(i+1) + '.png'
    sobel_name = 'SobelBaseline/'+ 'sobel_' + str(i+1) + '.png'

    img_name = "PBLite_"+str(i+1) + '.png'
    print('Finding boudaries for Image '+ name + '...')
    img = cv.imread('Images/'+ name + '.jpg')

    ## Making Tg:
    # print('Calculating Gradients')
    t_map = Texton_Map(img)
    T_g = []
    for map in t_map:
        Tg = Chi_Square(hd_bank,64,map)
        Tg = np.array(Tg)
        Tg = np.mean(Tg,axis = 0)
        plt.imsave('Results/Tg/Tg_'+name+'.png',Tg)
        # Tg = Tg - np.min(Tg)*np.ones(np.shape(Tg))
        # Tg = Tg/np.max(Tg)
        # print('Tg shape=',np.shape(Tg))
        T_g.append(Tg)

        # plt.imshow(Tg)
        # plt.show()
        # cv.imshow('Tg',Tg)
        # print(Tg)

    b_map = Brightness_Map(img)
    B_g = []
    for map in b_map:          
        Bg = Chi_Square(hd_bank,16,map)
        Bg = np.array(Bg)
        Bg = np.mean(Bg,axis = 0)
        B_g.append(Bg)
        plt.imsave('Results/Bg/Bg_'+name+'.png',Bg)

        # plt.imshow(Bg)
        # plt.show()

    #     # cv.imshow('Bg',Bg)
    #     # cv.waitKey(0)
    #     B_g.append(Bg)
        
    c_map = Color_Map(img)
    C_g = []
    for map in c_map:
        Cg = Chi_Square(hd_bank,16,map)
        Cg = np.array(Cg)
        Cg = np.mean(Cg,axis = 0)
        # cv.imshow('Bg',Cg)
        # cv.waitKey(0)
        C_g.append(Cg)
        plt.imsave('Results/Cg/Cg_'+name+'.png',Cg)

        
    canny = cv.imread(canny_name)
    sobel = cv.imread(sobel_name)

    # cv.imshow('canny',canny)
    # cv.waitKey(0)

    canny = cv.cvtColor(canny, cv.COLOR_BGR2GRAY)
    sobel = cv.cvtColor(sobel, cv.COLOR_BGR2GRAY)
    T1 = (T_g[0] + B_g[0] + C_g[0])/3
    w1 = 0.5
    w2 =0.5  
    T2 = w1*canny +w2*sobel
    edges = np.multiply(T1,T2)
    # gray_edge = cv.cvtColor(edges, cv.COLOR_BGR2GRAY)
    # plt.figure()
    plt.imshow(edges,cmap='gray')
    plt.axis('off')
    plt.imsave('Results/PB_Lite/'+ str(img_name),edges,cmap = 'gray')
    print('Saving the outputs for Image ' + name)
    # print('Done')
    # plt.savefig('Results/PB_Lite/'+ str(img_name))
    plt.show()

def Plot_n_Save():
    print('Saving Filter Bank Images')
    f_bank,odog,lms,lml,gb = filter_bank()
    plt.figure(figsize= (20,5),dpi = 80)
    for r,i in enumerate(odog):
        name = 'Oriented DOG Filter Bank'
        plt.subplot(2,16,r+1)
        plt.imshow(i,cmap='gray')
        plt.axis('off')
        # plt.suptitle(name,fontsize = 30)
    # plt.show()
    plt.savefig('Results/Filter_Banks/'+ name +'.png')

    plt.figure(figsize= (20,10),dpi = 80)
    for r,i in enumerate(lms):
        name = 'LM Small'

        plt.subplot(4,12,r+1)
        plt.imshow(i,cmap='gray')
        plt.axis('off')
        # plt.suptitle('LM Small',fontsize = 30)
    # plt.show()
    plt.savefig('Results/Filter_Banks/'+ name +'.png')


    plt.figure(figsize= (20,10),dpi = 80)
    for r,i in enumerate(lml):
        name = 'LM Large'

        plt.subplot(4,12,r+1)
        plt.imshow(i,cmap='gray')
        plt.axis('off')
        # plt.suptitle('LM Large',fontsize = 30)
    # plt.show()
    plt.savefig('Results/Filter_Banks/'+ name +'.png')

    plt.figure(figsize= (8,10),dpi = 100)
    for r,i in enumerate(gb):
        name = 'Gabor Filter'
        plt.subplot(8,6,r+1)
        plt.imshow(i,cmap='gray')
        plt.axis('off')
        # plt.suptitle('Gabor_Filter',fontsize = 30)     
    # plt.show()
    plt.savefig('Results/Filter_Banks/'+ name +'.png')

    hd_bank = Half_Disk_Bank()
    plt.figure(figsize= (20,10),dpi = 100)
    for r,i in enumerate(hd_bank):
        name = 'Half Disk'
        plt.subplot(6,8,r+1)
        plt.imshow(i,cmap='gray')
        plt.axis('off')
        # plt.suptitle('Gabor_Filter',fontsize = 30)     
    # plt.show()
    plt.savefig('Results/Filter_Banks/'+ name +'.png')

    print('All Done!')



Plot_n_Save()
