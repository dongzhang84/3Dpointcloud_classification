import laspy
import scipy
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import path
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

image_path = 'data_tanks/sample3.1.las'
image_save0 = 'data_tanks/sample3'

inFile = laspy.file.File(image_path)
dataset = np.vstack([inFile.x, inFile.y, inFile.z, inFile.red, inFile.green, inFile.blue]).transpose()
print('original dataset size', dataset.shape)

color = dataset[:,3:6]/65535
dataset1 = dataset[:,0:3]
dataset1[:,1] = dataset1[:,1]  * 0.2


max1 = dataset1[:,0].max()
min1 = dataset1[:,0].min()

max2 = dataset1[:,1].max()
min2 = dataset1[:,1].min()

max3 = dataset1[:,2].max()
min3 = dataset1[:,2].min()

# clustering!

clustering = DBSCAN(eps=0.24, min_samples=100, leaf_size=10).fit(dataset1)


core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
core_samples_mask[clustering.core_sample_indices_] = True
labels = clustering.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

unique_labels = set(labels)
classfication = []
for k in unique_labels:
    class_member_mask = (labels == k)
    xyz = dataset1[class_member_mask & core_samples_mask]
    classfication.append(len(xyz))
    
top_class = [classfication.index(x) for x in classfication if x>=0.1 * max(classfication)]
print(top_class)
top_number = len(top_class)


# OpenCV recognition

image_tot = [[0]]*len(top_class)
fig = [[0]]*len(top_class)

warnings.simplefilter("ignore", DeprecationWarning)


min_residual = 1.e8
pipe_index = 0

#fig = plt.figure()
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
  for each in np.linspace(0.3, 1, top_number)]

#for k, col in zip(unique_labels, colors):
for i,col in zip(range(0,top_number),colors):
    
    fig[i] = plt.figure(figsize=[8, 8])
    ax = fig[i].add_subplot(111)
    
    canvas = FigureCanvas(fig[i])
    
    k = top_class[i]
    class_member_mask = (labels == k)
    
    #print(k,class_member_mask)
    if k in top_class:
    #if k==4:
        xyz = dataset1[class_member_mask & core_samples_mask]
        
        XX = xyz[:,0]
        YY = xyz[:,1]
        ZZ = xyz[:,2]
        
        YY2D = YY
        
        XX_lenth = XX.max()-XX.min()
        YY_lenth = YY.max()-YY.min()
        ZZ_lenth = ZZ.max()-ZZ.min()
        
        ax.scatter(XX, YY2D, c='k')
        # Turn off tick labels
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.axis('off')
        
        plt.close()
        
        canvas.draw()     
        
        #print(i)
        
        image = np.fromstring(fig[i].canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(fig[i].canvas.get_width_height()[::-1] + (3,))
        image_tot[i] = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

################ Find Tanks ########################

sample_index = 0

# the sample image which is confirmed to be the 2D projection of a tank
img_sample = image_tot[sample_index]

# to select the clustering indices which are tanks
tank_index = []
tank_index.append(sample_index)

for i in range(0,len(top_class)):
    
    img_compare = image_tot[i]
    
    orb = cv2.ORB_create()
    
    kp_a, desc_a = orb.detectAndCompute(img_sample, None)
    kp_b, desc_b = orb.detectAndCompute(img_compare, None)
    
    # initialize the bruteforce matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # match.distance is a float between {0:100} - lower means more similar
    matches = bf.match(desc_a, desc_b)
    similar_regions = [i for i in matches if i.distance < 40]
    if len(matches) == 0:
        print(0)
    score = len(similar_regions)/len(matches) 
    print(top_class[i],score)
    if (score > 0.1) & (i not in tank_index) :
        tank_index.append(top_class[i])
    
print(tank_index)


################### Plot Tanks #############################

fig = plt.figure(figsize=[50, 20])
ax = fig.add_subplot(111, projection='3d')

# This array to store tanks' position, tank_data[i] is the ith tank
tank_data = [[0]]*len(tank_index)
image_save = [[]]*len(tank_index)
MM = 0

colors = [plt.cm.Spectral(each)
  for each in np.linspace(0.3, 1, top_number)]

#print(colors)

for i,col in zip(range(0,top_number),colors):
    k = top_class[i]
    #print(k,col)
    class_member_mask = (labels == k)
    
    if k in tank_index:
    #if k == 1:
        xyz = dataset1[class_member_mask & core_samples_mask]
        col_pipe = color[class_member_mask & core_samples_mask]
        print(k,len(xyz))
        xyz[:, 1] = xyz[:, 1]/0.2
        tank_data[MM] = xyz
        
        image_save[MM] = image_save0+'.'+str(MM)+'.csv'
        
        tank_name = 'tank_csv'+str(MM)
        
        tank_name = np.concatenate((tank_data[MM], col_pipe * 255), axis=1)
        pd.DataFrame(tank_name).to_csv(image_save[MM])

        MM += 1
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=col_pipe, marker=".")

plt.show()




