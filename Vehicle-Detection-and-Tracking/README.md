##Vehicle Detection Project

![alt text][gif1]

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use trained SVM classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[gif1]: ./output_images/project_video_output.gif
[image1]: ./output_images/vehicle.png
[image2]: ./output_images/non_vehicle.png
[image3]: ./output_images/hog.jpg
[image4]: ./output_images/Multi_scale_windows.jpg
[image5]: ./output_images/Sliding_window_SVM_classify.jpg
[image6]: ./output_images/Centroid_window_heatmap.jpg
[video1]: ./project_video_output.mp4

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook `Vehicle-Detection-and-Tracking.ipynb`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]  ![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HLS` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image3]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters such as 

* `Color space` = RGB, HSV, LUV, HLS, YUV, YCrCb
* `orientations` = 8, 9, 10, 11, 12
* `pixels_per_cell` = (8, 8)
* `cells_per_block`= (2, 2), (4, 4)

I selected the combination that gave best SVM test accuracy of 99.13%, which is
`color space=HLS`, `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in seventh code cell of `Vehicle-Detection-and-Tracking.ipynb` using combination of both color-based and shape based features. We perform spatial binning on an image and still retain enough information to help in finding vehicles. We also compute color histogram features. Color-based features only capture one aspect of object's appearance. When we have class of objects that can vary in color, gradients will give more robust representation. We compute histogram of gradient directions or orientations feature.

All the above features are combined by concatinating them. Features are normalized after concatination to avoid one type of feature dominating the other.

That is fed to Linear SVM classifier.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In general, we don't know what size our object of interest will be in the image we are searching. So it makes sense to search multiple scales. Here we are establishing minimum and maximum scale at which we expect the object to appear and then reasonable number of intermediate scales to scan as well.

We have restricted the search area for the cars in an image to lower half of the image as shown below. Also window sizes is chosen by taking into account that cars appear smaller near the horizon and larger when they are close to our camera.

After so many iterations of traial and error. I chose the following window sizes with 80% overlap for all

| Window size   | Y\_start\_stop|
| ------------- | ------------- |
|   (90, 58)    |  (400, 530)   |
|   (110, 71)   |  (400, 550)   |
|   (130, 84)   |  (400, 570)   |
|   (150, 97)   |  (400, 590)   |
|   (170, 110)  |  (400, 610)   |
|   (190, 123)  |  (400, 630)   |

![alt text][image4]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using HLS 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image6]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Even after using heatmaps to remove false positives, there were few false positives present, and I had hard time getting rid of them. Finally, I found out a trick for that, which is, to threshold the output of `LinearSVC.decision_function()` whenever `LinearSVC.predict()` predicts the Car class in any patch. This eliminated the false positives that could not be filtered by heatmap.

