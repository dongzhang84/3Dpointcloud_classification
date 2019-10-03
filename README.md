# 3D Point Cloud Images Segmentation and Classification

This is my [Insight Data Science](https://www.insightdatascience.com/) project which can benefit energy industry companies, and potentially all 3D computer vision-related industry. Energy industry companies spend billions of dollars to do asset inspection and maintenance each year. What they are doing now is to use robotic systems (e.g., drones) to take 2D pictures of their assets, and use photogrammetry algorithms to generate 3D images for inspection ([more details](https://www.youtube.com/watch?v=ZwqNn1x7OoE)). In order to better understand the properties of various components such as pipes, valves, tanks and so on, it is crucial to do real-time 3D image segmentation and classification, which are currently time-consuming tasks and often processed manually.

Thus, I created a tool to segment 3D objects and classify some of them automatically, using unsupervised machine learning clustering and computer vision techniques. 

In addition to the github repo, I also describe the details of my project in the website:

https://dongzhang84.github.io/classify3d

My presentation slides and video can be given upon request.
 


## Introduction

- The 3D images processed by my tool are in Point Cloud las format ([more details](https://en.wikipedia.org/wiki/Point_cloud)). Each image is 2-10 GB with 50-100 million points including (x, y, z) coordinates and RGB color information. To prepare for machine learning and computer vision algorithms, I artificially cut the images to small "local" region, reducing the spatial point resolution so that each "local" 3D image contains 1 to 2 million points, which can be easily handled by a personal laptop. 

- This tool was wrote in Python. Point cloud data is read by Python using package laspy. Machine learning tool scikit-learn and openCV are installed in Python. In particular, my project is to automatic identify pipes and tanks in point cloud images of water treatment plans, which asset components are comparable to oil and gas infrastructure. 

- The major workflow with detailed instruction can be seen in [Automatic_Pipe_Finder](https://github.com/dongzhang84/3Dpointcloud_classification/blob/master/automatic_pipe_finder.ipynb). Another workflow for tank identification can be found here [Automatic_Tank_Finder](https://github.com/dongzhang84/3Dpointcloud_classification/blob/master/automatic_tank_finder.ipynb). In general the workflow can be divided into four parts: data loading and pre-processing, image segmentation using DBSCAN clustering, data classification using openCV (ORB), saving the selected object(s) as csv files, and finally, validation. 

- The pipelines for automatic 3D segmentation and classification are written in python, including the pipeline for automatic 3D pipe finder ([automatic_pipe_finder.py](https://github.com/dongzhang84/3Dpointcloud_classification/blob/master/automatic_pipe_finder.py)), and automatic 3D tank finder ([automatic_tank_finder.py](https://github.com/dongzhang84/3Dpointcloud_classification/blob/master/automatic_tank_finder.py)). Note that the tank finder is still in beta version and under test. 

- The web application of 3D pipe finder is in Flask. Please run app.py locally. Some instruction to use my pipelines and Flask app are as follows.

## Instruction

The python pipelines have been tested with Python 3.7.4. 

- To run [automatic_pipe_finder.py](https://github.com/dongzhang84/3Dpointcloud_classification/blob/master/automatic_pipe_finder.py), and for 3D tank finder [automatic_tank_finder.py](https://github.com/dongzhang84/3Dpointcloud_classification/blob/master/automatic_tank_finder.py), one needs to first set up the directories image_path and image_save at the beginning of code, for example, set
 
        image_path = 'data_pipes/data_test1/sample3.las'
        image_save = 'data_pipes/data_test1/sample3.csv'

The image_path is the directory of point cloud file you want to open for processing, and the image_save directory saves the selected pipe(s). Then one can run the automation pipeline using the following command to find and save pipe(s). Very similar process can be performed for 3D tank finder. 

        $ python3 automatic_pipe_finder.py

- The web application of 3D pipe finder is deployed by Flask. The web app was built locally, one can run it on one's own computer. To use the web application, one needs to change the directories in Flask/app.py to the directories to read and save files:

        image_path0 = '../../Hybird_data_codes/data_pipes/data_test/'
        image_save0 = '../../Hybird_data_codes/data_pipes/data_test/'

 After setting up the directories, in Flask's home directory run

         $ python3 app.py

and open the web application in your local browser: http://0.0.0.0:8000/. The web looks like:
![prediction example](https://github.com/dongzhang84/3Dpointcloud_classification/blob/master/Flask/figures/web1.png)

Select a file and upload. After a couple of minute processing, the web should load the done page as below, then you can find the automatically selected pipes has been saved in the directory image_save0 as a csv file.

![prediction example](https://github.com/dongzhang84/3Dpointcloud_classification/blob/master/Flask/figures/web2.png)


## Delivered Summary

Using my tool described above, I help energy industry companies to better use their 3D images. In particular, currently 3D asset component segmentation and labeling are pretty time-consuming. For example, for a 3D image of water treatment factory with 50 million points, it takes a person 3-4 hours to label pipes in the image. My tool can provide an automatic way to select pipes, and tanks. Potentially this technique may not only be implemented to do real-time inspection for energy assets, but for all 3D computer-vision-related industry. 
