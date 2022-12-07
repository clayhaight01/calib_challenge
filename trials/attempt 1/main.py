from tkinter import N
import numpy as np
from cmath import inf
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
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
WDW_LEN = 5
file_path = "C:/Users/User/Documents/GitHub/calib_challenge/unlabeled/"

# Frame counts
# NUM_FRAMES = len(os.listdir(file_path+"5"))
NUM_FRAMES = 10
imgs = []
kp = [None] * NUM_FRAMES
des = [None] * NUM_FRAMES
matches = [None] * (NUM_FRAMES - 1)
links = []



cmap = plt.cm.cool
norm = plt.Normalize(vmin=0, vmax=50)
scalarMap = cmx.ScalarMappable(norm=norm,cmap=cmap)

# Compute individual images
for i in range(NUM_FRAMES-1):
    imgs.append(cv.imread(file_path+"5/"+f"frame{i}.jpg",cv.IMREAD_GRAYSCALE))
    kp[i], des[i] = orb.detectAndCompute(imgs[i],None)
img_height, img_width = imgs[0].shape[:2]


bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Compute links between images
for i in range(NUM_FRAMES-1): #range(len(imgs)-1):
    delta_list = []
    hamming_list = []
    line_center = []
    links.append([])
    matches[i] = bf.match(des[i],des[i+1]) # query then train
    matches[i] = sorted(matches[i], key = lambda x:x.distance)

    # Populate links, and hamming/delta lists
    for j in range(len(matches[i])):
        links[i].append(Link(matches[i][j], kp[i][matches[i][j].queryIdx], kp[i+1][matches[i][j].trainIdx]))
        hamming_list.append(links[i][j].hamming)
        delta_list.append(links[i][j].delta)
    
    figure, axis = plt.subplots(3, gridspec_kw={'height_ratios': [5, 1, 1]})
    axis[0].imshow(imgs[i])
    axis[0].set_xlim(0, img_width)
    axis[0].set_ylim(img_height, 0)
    axis[0].set_title(f"Frame {i}")

    axis[1].plot(range(len(delta_list)),delta_list,'navy')
    axis[2].plot(range(len(hamming_list)),hamming_list,'y')

    # Filter links based on heuristics
    filtered = []
    for j in range(len(links[i])):
        if delta_list[j] < 3 or (delta_list[j] > (np.mean(delta_list)+2*np.std(delta_list))) or (hamming_list[j] > (np.mean(hamming_list)+np.std(hamming_list))):
            filtered.append(j)
            axis[1].axvline(x=j)
            axis[2].axvline(x=j)
    filtered.reverse()

    for k in filtered:
        del links[i][k]
    
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
            vector_intersects_x.append(point_intersect[0])
            vector_intersects_y.append(point_intersect[1])
            # axis[0].add_patch(plt.Circle(point_intersect,1,color='r'))

        # Filter out intersections of picture, and out of 0.1 std dev
        if f > 1:
            num_del = 0
            for h1 in range(len(vector_intersects_x)-num_del):
                if 0>vector_intersects_x[h1-num_del]>img_width or 0>vector_intersects_y[h1-num_del]>img_height:
                    del vector_intersects_x[h1-num_del]
                    del vector_intersects_y[h1-num_del]
                    num_del+=1
            vector_intersects_x_med = sorted(vector_intersects_x)[len(vector_intersects_x)//2]
            vector_intersects_y_med = sorted(vector_intersects_y)[len(vector_intersects_y)//2]
            for h2 in range(len(vector_intersects_x)-num_del):
                if vector_intersects_x[h2-num_del]>vector_intersects_x_med + 0.1*np.std(vector_intersects_x) or vector_intersects_y[h2-num_del]>vector_intersects_y_med + 0.1*np.std(vector_intersects_y):
                    del vector_intersects_x[h2-num_del]
                    del vector_intersects_y[h2-num_del]
                    num_del+=1

            vector_intersects_x_avg = sum(vector_intersects_x)/len(vector_intersects_x)
            vector_intersects_y_avg = sum(vector_intersects_y)/len(vector_intersects_y)
            line_center.append([vector_intersects_x_avg,vector_intersects_y_avg])
            axis[0].add_patch(plt.Circle([vector_intersects_x_avg,vector_intersects_y_avg],1,color='b'))

        #axis[0].add_patch(plt.Circle(400,500,1,color='r'))

        colorVal = scalarMap.to_rgba(links[i][f].hamming)
        axis[0].arrow(pt_curr[0], pt_curr[1], 5*(pt_next[0]-pt_curr[0]), 5*(pt_next[1]-pt_curr[1]), head_width=2, head_length=3, fc=colorVal, ec=colorVal)
        #axis[0].arrow(pt_curr[0]-25*(pt_next[0]-pt_curr[0]), pt_curr[1]-25*(pt_next[1]-pt_curr[1]), 50*(pt_next[0]-pt_curr[0]), 50*(pt_next[1]-pt_curr[1]), head_width=2, head_length=3, fc=colorVal, ec=colorVal)
    travel_center = [0,0]
    for m in range(len(line_center[0])):
        travel_center[0] += line_center[m][0]
        travel_center[1] += line_center[m][1]
    travel_center[0]/=len(line_center[0])
    travel_center[1]/=len(line_center[1])
    print(travel_center)
    axis[0].add_patch(plt.Circle(travel_center,2,color='y'))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    figure.colorbar(sm, None, axis[0], True)
    plt.show()
