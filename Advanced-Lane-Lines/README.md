## Advanced Lane Finding

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[gif1]: ./output_images/project_video_output.gif "Final output"
[image1]: ./output_images/1_camera_calibration.png  "image1"
[image2]: ./output_images/2_undistorted_raw_road_image.png  "image2"
[image3]: ./output_images/3_thresholded_binary_output.png  "image3"
[image4]: ./output_images/4_warped_image.png  "image4"
[image5]: ./output_images/5_lines_on_warped_image.png  "image5"
[image6]: ./output_images/6_lane_marked_on_original_image.png  "image6"

![alt text][gif1]
## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in `Advanced_lane_finding.ipynb`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of both the x and y gradients that meet the threshold criteria, or the gradient magnitude and direction that are both within their threshold values, or selection for pixels where both the h and s color channel s that meet the threshold criteria to generate a binary image (thresholding steps at cell 5 in `Advanced_lane_finding.ipynb `).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image()`, which appears in cell 7 in the file `Advanced_lane_finding.ipynb`.  The `warp_image()` function takes as inputs an image (`image `), as well as camera calibration matrices `mtx`, `dist`, and `M`.  I chose the hardcode the source and destination points in the following manner:

```
 x = image.shape[1]
 y = image.shape[0]
 src = np.float32([[(x/2 - 150, y/2 + 100),
                    (x/2 + 150, y/2 + 100),
                    (x - 50,    y - 40),
                    (100,       y - 40)]])
 dst = np.float32([[(0,      0),
                    (x - 50, 0),
                    (x - 50, y),
                    (100,    y)]])
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 490, 460      | 0, 0          | 
| 790, 460      | 1230, 0       |
| 1230, 680     | 1230, 720     |
| 100, 680      | 100, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Line Finding Method: Peaks in a Histogram and Sliding Window.
With this histogram I am adding up the pixel values along each column in the image. In my thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. I used that as a starting point for where to search for the lines. From that point, I used a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame, and then fit a second order polynomial to find left and right lane pixels.

Code to identify lane-line pixels is in cell 9 of `Advanced_lane_finding.ipynb`
Then I fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Lane Curvature:
The above found x and y coordinates of lane pixels are fit a new polynomial for each line to the real-world sized data. Using this new polynomial, I used radius of curvature formula given ([here](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/2f928913-21f6-4611-9055-01744acc344f)) on both left and right lanes

Lane Position:
I find the bottom postion of both left and right lanes. Assuming the actual lane width of 3.7 meters, I used the distance between center of lane and center of image to find the offset of the car.

I did this in cell 9 and 11 in `Advanced_lane_finding.ipynb `

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell 12 in `Advanced_lane_finding.ipynb `.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problem I faced was to figure of the right mask for the white lanes in different lighting conditions and different road surface color. Finally I ended up using a combination of both the x and y gradients that meet the threshold criteria, or the gradient magnitude and direction that are both within their threshold values, or selection for pixels where both the h and s color channel s that meet the threshold criteria to generate a binary image.

My pipeline will fail on the two challenge videos provided, `challenge_video.mp4` and `harder_challenge_video.mp4`. That's because `challenge_video.mp4` has a huge crack line right in the center of the lanes, which could be easily mistaken for an actual lane. Also `harder_challenge_video.mp4` has so many curved roads, I need a different pipeline to handle that.

