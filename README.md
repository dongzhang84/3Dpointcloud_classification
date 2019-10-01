# 3D Point Cloud Images Segmentation and Classification

This is my Insight Data Science project which can benefit energy industry companies, and potentially all 3D computer vision-related industry. Industrial energy asset companies spend billions of dollars per year to do inspection and maintenance. What they are doing now is to use robotic systems (e.g., drones) to take 2D pictures of their assets and photogrammetry algorithms to generate 3D images for inspection ([more details](https://www.youtube.com/watch?v=ZwqNn1x7OoE)). In order to better understand the properties of various components such as pipes, valves, tanks and so on, it is crucial to do real-time 3D image segmentation and classification, which are unfortunately handled manually and pretty time-consuming in many cases. I created a tool to segment 3D objects and classify some of them automatically using unsupervised machine learning clustering and computer vision techniques. 

## Deliverable Summary 

- The 3D images are in Point Cloud las format ([more details](https://en.wikipedia.org/wiki/Point_cloud)). Each image is 2-10 GB with 50-100 million points including (x, y, z) coordinates, and RGB color information. To prepare for machine learning and computer vision algorithms, I artificially cut the images to small "local" region, or reduced the spatial point resolution so each "local" 3D image contains 1 to 2 million points. 

- The point cloud data is loaded in Python using package laspy. All the work was done using Python. The workflow can be seen in this [Jupyter notebook](https://github.com/dongzhang84/3Dpointcloud_classification/blob/master/automatic_pipe_finder.ipynb). In particular, my project is to identify pipes and tanks in point cloud images of water treatment plans, which asset components are comparable to oil and gas infrastructure. 

- The pipelines for automatic segmentation and classification are written in python, including the pipeline for 3D pipe finder ([automatic_pipe_finder.py](https://github.com/dongzhang84/3Dpointcloud_classification/blob/master/automatic_pipe_finder.ipynb)), and for 3D tank finder ([automatic_tank_finder.py](https://github.com/dongzhang84/3Dpointcloud_classification/blob/master/automatic_pipe_finder.ipynb)). Note that the tank finder is still in the beta version. 

To run the python code, one needs to first set up the image_path and image_save in the python code, for example

image_path = 'data_pipes/data_test1/sample3.las'
image_save = 'data_pipes/data_test1/sample3.csv'

The image_path is the directory of point cloud file you want to open and process, and the image_save directory saves the selected pipe(s). Then one can run 

$ python3 automatic_tank_finder.py

to find and save pipe(s). The same process can be perform for 3D tank finder. 


- The web application of 3D pipe finder is in Flask. To use the web application, one needs to change the directories in app.py:

image_path0 = '../../Hybird_data_codes/data_pipes/data_test/'
image_save0 = '../../Hybird_data_codes/data_pipes/data_test/'

to the directories one wants to read and save files. Then run

python3 app.py

and open the web application in your local browser: http://0.0.0.0:8000/

select a file and upload, after data processing, the web interface becomes like this,

and you can find the automatically selected pipes in the directory image_save0. 