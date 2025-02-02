from tkinter import N
import numpy as np
import cmath
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from tqdm import tqdm
import time
import os

class Link:
    def __init__(self, match, kp1, kp2):
        self.hamming = match.distance
        self.kp_curr = kp1
        self.kp_next = kp2
        self.delta = np.linalg.norm(np.subtract(self.kp_next.pt, self.kp_curr.pt))

    def __str__(self):
        return f"kp1: ({self.kp_curr.pt[0]:.2f}, {self.kp_curr.pt[1]:.2f}) kp2: ({self.kp_next.pt[0]:.2f}, {self.kp_next.pt[1]:.2f}) quality: {self.hamming:.2f} distance: {self.delta:.2f}"

    def calculate_quality(self, hamming_list, delta_list):
        u_hamming = np.mean(hamming_list)
        o_hamming = np.std(hamming_list)
        u_delta = np.mean(delta_list)
        o_delta = np.std(delta_list)
        return

def get_slope(p1, p2):
    if p2[0]==p1[0]:
        return 0
    return (p2[1]-p1[1])/(p2[0]-p1[0])

def intersect(p1, m1, p2, m2):
    if m1 == m2:
        return [float("Nan"), float("Nan")]
    p3 = [0,0]
    p3[0]=int((m1*p1[0]-m2*p2[0]-p1[1]+p2[1])/(m1-m2))
    p3[1]=int(m1*p3[0]+p1[1]-m1*p1[0])
    return p3

orb = cv.ORB_create()
file_path = "C:/Users/User/Documents/GitHub/calib_challenge/unlabeled/"

TST_FRAMES = 1200
F_LENGTH = 910
GRID_SZ = 
imgs = []
kp = [None] * TST_FRAMES
des = [None] * TST_FRAMES
matches = [None] * (TST_FRAMES - 1)
links = []



cmap = plt.cm.cool
norm = plt.Normalize(vmin=0, vmax=50)
scalarMap = cmx.ScalarMappable(norm=norm,cmap=cmap)

# Compute individual images
l_time = time.time()
print("Finding Keypoints...")
for i in tqdm(range(TST_FRAMES-1)):
    imgs.append(cv.imread(file_path+"5/"+f"frame{i}.jpg",cv.IMREAD_GRAYSCALE))
    kp[i], des[i] = orb.detectAndCompute(imgs[i],None)
img_height, img_width = imgs[0].shape[:2]
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
print(time.time()-l_time)
l_time=time.time()

# Compute links between images
for i in range(TST_FRAMES-1): #range(len(imgs)-1):
    delta_list = []
    hamming_list = []
    line_center = []
    links.append([])
    print("Matching Keypoints...")
    matches[i] = bf.match(des[i],des[i+1]) # query then train
    matches[i] = sorted(matches[i], key = lambda x:x.distance)

    # Populate links, and hamming/delta lists
    l_time = time.time()
    print("Populating Lists...")
    for j in range(len(matches[i])):
        links[i].append(Link(matches[i][j], kp[i][matches[i][j].queryIdx], kp[i+1][matches[i][j].trainIdx]))
        hamming_list.append(links[i][j].hamming)
        delta_list.append(links[i][j].delta)
    print(time.time()-l_time)
    l_time=time.time()

    print("Plotting...")
    figure, axis = plt.subplots(3, gridspec_kw={'height_ratios': [10, 1, 1]})
    
    axis[0].imshow(imgs[i])
    axis[0].set_xlim(0, img_width)
    axis[0].set_ylim(img_height, 0)
    axis[0].set_title(f"Frame {i}")

    axis[1].plot(range(len(delta_list)),delta_list,'navy')
    axis[2].plot(range(len(hamming_list)),hamming_list,'y')
    print(time.time()-l_time)
    l_time=time.time()

    # Filter links based on heuristics
    num_del = 0
    for j in range(len(links[i])-num_del):
        if delta_list[j] < 3 or (delta_list[j] > (np.mean(delta_list)+np.std(delta_list))) or (hamming_list[j] > (np.mean(hamming_list)+np.std(hamming_list))):
            axis[1].axvline(x=j)
            axis[2].axvline(x=j)
            del links[i][j-num_del]
            num_del += 1
    
    # Plot arrows and intersections
    for f in range(len(links[i])): 
        pt_curr = links[i][f].kp_curr.pt
        pt_next = links[i][f].kp_next.pt

        m_link = get_slope(pt_curr, pt_next)

        vector_intersects_x = []
        vector_intersects_y = []
        for g in range(f-1):
            tpt_curr = links[i][g].kp_curr.pt
            tpt_next = links[i][g].kp_next.pt
            t_m = get_slope(tpt_curr, tpt_next)
            point_intersect = intersect(pt_curr, m_link, tpt_curr, t_m)
            if not cmath.isnan(point_intersect[0]):
                vector_intersects_x.append(point_intersect[0])
                vector_intersects_y.append(point_intersect[1])
                #axis[0].add_patch(plt.Circle(point_intersect,1.5,color='r'))

        # Filter out intersections of picture
        if len(vector_intersects_x)>0 and f>1:# and len(vector_intersects_x)>2:
            num_del = 0
            for h1 in range(len(vector_intersects_x)-num_del):
                if vector_intersects_x[h1-num_del]<0 or vector_intersects_x[h1-num_del]>img_width or vector_intersects_y[h1-num_del]<0 or vector_intersects_y[h1-num_del]>img_height:
                    del vector_intersects_x[h1-num_del]
                    del vector_intersects_y[h1-num_del]
                    num_del+=1
            x_med = sorted(vector_intersects_x)[len(vector_intersects_x)//2]
            y_med = sorted(vector_intersects_y)[len(vector_intersects_y)//2]
            for h2 in range(len(vector_intersects_x)-num_del):
                cutoff_std = 2
                x_curr = vector_intersects_x[h2-num_del]
                y_curr = vector_intersects_y[h2-num_del]
                x_std = cutoff_std * np.std(vector_intersects_x)
                y_std = cutoff_std * np.std(vector_intersects_y)
                if x_curr>x_med + x_std or x_curr<x_med - x_std or y_curr>y_med + y_std or y_curr<y_med - y_std:
                    del vector_intersects_x[h2-num_del]
                    del vector_intersects_y[h2-num_del]
                    num_del+=1

            vector_intersects_x_avg = sum(vector_intersects_x)/len(vector_intersects_x)
            vector_intersects_y_avg = sum(vector_intersects_y)/len(vector_intersects_y)
            line_center.append([vector_intersects_x_avg,vector_intersects_y_avg])
            axis[0].add_patch(plt.Circle([vector_intersects_x_avg,vector_intersects_y_avg],2,color='r'))

        colorVal = scalarMap.to_rgba(links[i][f].hamming)
        axis[0].arrow(pt_curr[0], pt_curr[1], 1*(pt_next[0]-pt_curr[0]), 1*(pt_next[1]-pt_curr[1]), head_width=2, head_length=3, fc=colorVal, ec=colorVal)
    
    travel_center = [0,0]
    for m in range(len(line_center)):
        travel_center[0] += line_center[m][0]
        travel_center[1] += line_center[m][1]
    travel_center[0]/=len(line_center)
    travel_center[1]/=len(line_center)
    crc_vector = [travel_center[0]-img_width//2,travel_center[1]-img_height//2]
    axis[0].arrow(img_width//2, img_height//2, crc_vector[0], crc_vector[1], head_width=2, head_length=3, fc='lime', ec='lime')

    c_angles = [cmath.atan(crc_vector[1]/F_LENGTH), cmath.atan(crc_vector[0]/F_LENGTH)]
    print(travel_center, c_angles)
    axis[0].add_patch(plt.Circle(travel_center,3,color='y'))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    figure.colorbar(sm, None, axis[0], True)
    plt.show()
