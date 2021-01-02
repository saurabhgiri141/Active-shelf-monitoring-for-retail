## Active-shelf-monitoring-for-retail
The application of the new version of YOLO is introduced here, i.e. YOLOv5, to classify items on the shelf in a retail store. This program can be used to keep track of product inventory simply by using images of the products on the shelf.

![Result image](https://github.com/saurabhgiri141/Active-shelf-monitoring-for-retail/blob/main/results.png)

## Introduction
Object detection is a computer vision task that requires object(s) to be detected, localized and classified. In this task, first we need our machine learning model to tell if any object of interest is present in the image. If present, then draw a bounding box around the object(s) present in the image. In the end, the model must classify the object represented by the bounding box. This task requires fast object detection so that it can be implemented  in real-time. One of its major applications is its use in real-time object detection in self-driving vehicles.

## Objective
To use YOLOv5 to draw bounding boxes over retail products in pictures using SKU110k dataset.

## Dataset
To do this task, first I downloaded the  SKU110k image dataset from the following link: 
http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz
The SKU110k dataset is based on images of retail objects in a densely packed setting. It provides training, validation and test set images and the corresponding .csv files which contain information for bounding box locations of all objects in those images. The .csv files have object bounding box information written in the following columns: 

image_name,x1, y1, x2, y2, class, image_width, image_height

where x1,y1 are top left co-ordinates of bounding box and x2, y2 are bottom right co-ordinates of bounding box, rest of parameters are self-explanatory. An example of parameters of train_0.jpg image for one bounding box, is shown below. There are several bounding boxes for each image, one box for each object.

train_0.jpg, 208, 537, 422, 814, object, 3024, 3024

In the SKU110k dataset, we have 2940 images in the test set, 8232 images in the train set and 587 images in the validation set. Each image can have varying number of objects, hence, varying number of bounding boxes.


## Methodology
We only took 998 images from the training set from the dataset and went to the Roboflow.ai website, which offers an online image annotation service in various formats, including the format provided by YOLOv5. The explanation for only choosing 998 images from the training set is that the image annotation service of Roboflow.ai is free only for the first 1000 images.

### Preprocessing
Preprocessing of images includes resizing them to 416x416x3. This is done on Roboflow's platform. An annotated, resized image is shown in figure below:

![Annotated image](https://github.com/saurabhgiri141/Active-shelf-monitoring-for-retail/blob/main/roboflow_data_image_annotated.jpg)

Fig 1.3: Image annotated by Roboflow

### Automatic Annotation
The bounding box annotation .csv file and training set images are uploaded to the Roboflow.ai website, and Roboflow.ai's annotation service automatically draws bounding boxes on images using the annotations contained in the .csv files as seen in the above picture.

### Data Generation
Roboflow also gives option to generate a dataset based on user defined split. I used 70–20–10 training-validation-test set split. After the data is generated on Roboflow, we get the original images as well as all bounding box locations for all annotated objects in a separate text file for each image, which is convenient. Finally, we get a link to download the generated data with label files. This link contains a key that is restricted to only your account and is not supposed to be shared.

## Code
The code is present in jupyter notebook in attached files. However, it is recommended to copy the whole code in Google Colab notebook.

It is originally trained for COCO dataset but can be tweaked for custom tasks which is what I did. I started by cloning YOLOv5 and installing the dependencies mentioned in requirements.txt file. Also, the model is built for Pytorch, so I import that.

```
!git clone https://github.com/ultralytics/yolov5  # clone repo
!pip install -r yolov5/requirements.txt  # install dependencies
%cd yolov5
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
```

First, the dataset we generated at Roboflow.ai is downloaded. Training, test and validation sets and annotations can also be downloaded from the following code. It also generates a .yaml file containing training and validation set paths as well as what classes are present in our knowledge. If you use Roboflow for info, as it is special per user, don't forget to enter the key in the code.

```
# Export code snippet and paste here
%cd /content
!curl -L "ADD THE KEY OBTAINED FROM ROBOFLOW" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

This file tells the model the location path of training and validation set images alongwith the number of classes and the names of classes. For this task, number of classes is "1" and the name of class is "object" as we are only looking to predict bounding boxes. data.yaml file can be seen above.


## Observations
We can visualize important evaluation metrics after the model has been trained using the following code:

```
# we can also output some older school graphs if the tensor board isn't working for whatever reason... 
from utils.utils import plot_results; plot_results()  # plot results.txt as results.png
Image(filename='./results.png', width=1000)  # view results.png
```

The following 3 parameters are commonly used for object detection tasks:
· GIoU is the Generalized Intersection over Union which tells how close to the ground truth our bounding box is.
· Objectness shows the probability that an object exists in an image. Here it is used as loss function.
· mAP is the mean Average Precision telling how correct are our bounding box predictions on average. It is area under curve of precision-recall curve.
It is seen that Generalized Intersection over Union (GIoU) loss and objectness loss decrease both for training and validation. Mean Average Precision (mAP) however is at 0.7 for bounding box IoU threshold of 0.5. Recall stands at 0.8 as shown below:

![Observations](https://github.com/saurabhgiri141/Active-shelf-monitoring-for-retail/blob/main/observations.png)

Fig 1.4: Observations of important parameters of model training

Now comes the part where we check how our model is doing on test set images using the following code:
```
# when we ran this, we saw .007 second inference time. That is 140 FPS on a TESLA P100!
%cd /content/yolov5/
!python detect.py --weights weights/last_yolov5s_results.pt --img 416 --conf 0.4 --source ../test/images
```

## Results
Following images show the result of our YOLOv5 algorithm trained to draw bounding boxes on objects. The results are pretty good.

![results1](https://github.com/saurabhgiri141/Active-shelf-monitoring-for-retail/blob/main/download.jpg)

![results](https://github.com/saurabhgiri141/Active-shelf-monitoring-for-retail/blob/main/download%20(1).jpg)

## Conclusion
YOLOv5 performs well and can be customized to suit our needs. However, training the model can take significant GPU power and time. It is recommended to use atleast Google Colab with 16GB GPU or preferably a TPU to speed up the process for training the large dataset.

This retail object detector application can be used to keep track of store shelf inventory or for a smart store concept where people pick stuff and get automatically charged for it. YOLOv5's small weight size and good frame rate will pave its way to be first choice for embedded-system based real-time object detection tasks.
