
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: Results/HogFeatures.png
[image2]: Results/Scale1.png
[image3]: Results/Scale15.png
[image4]: Results/Scale2.png
[image5]: Results/Scale25.png
[image6]: Results/Scale3.png
[image7]: Results/TestExample5.png
[image8]: Results/TestExample1.png
[image9]: Results/TestExample2.png
[image10]: Results/TestExample3.png
[image11]: Results/TestExample4.png
[image12]: Results/Carposition1.png
[image13]: Results/Carposition2.png
[image14]: Results/Carposition3.png
[image15]: Results/Carposition4.png
[image16]: Results/Carposition5.png
[image17]: Results/Carposition6.png
[image18]: Results/Fail1.png
[image19]: Results/Fail2.png
[image20]: Results/Fail3.png
[video1]:  Results/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points


---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The feature extraction is defiend under "Define feature extraction functions" title of the notebook. I started with all three methods: spatial color, color histogram and HOG. The HOG feature extraction is defined in funtion "get_hog_features".

The HOG feature can be extracted from single channel or ALL chanels as listed in the "extract_features" function.

For the training data, I use all `non-vehicle` and `vehicle` data except the GTI_far and GTI_Left, becasue the data in these 2 folders doesnt really related to our video data. I started by reading in all the `vehicle` and `non-vehicle` images.  then played with the different skimage.hog parametes. I use default for `cells per block` and change the `orenetation` and `pexels_per_cell` to get a feel of what the output looks like.


Here is an example using the `YUV` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(10,10)` and `cells_per_block=(2, 2)`:

![alt text][image1]



#### 2. Explain how you settled on your final choice of HOG parameters.

As mentioned before, I first settled the `cells_per_block=(2,2)` as default, but 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM, the code is under tile of Training. I use `test_train_split` to define the 20% of the data as test data.
Initially I tried to use all three features ( spatical color, color histogram and HOG), and I used `StandardScaler()` function to normalize the data. I do get higher accuracy for the traning data, but it's not stable for the test video data. Then I just focus on the HOG feature as project required, where I realized I am getting better test results without data normalization. Since I am only using one type (HOG) feature, the normalization is not needed anyway. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in function `find_cars()`. After all the parametes is settled, I used `get_hog_features()` function to extract the HOG feature of each channel individually, then the sub sampling is implemented to extract feature for each search window. 
For this project, I used mutiple scales (1~3) to search. The smaller scale (1~2) is employed on the top part of the search area (380~650 pixels), and the larger scales are employed on the bottome part of the search area.

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

From the above result, the scale larger than 2 doesn't really help on dectection. So I only implement 1,1.5 and 2 in the following study.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on the above 3 scales using YUV 3-channel HOG features. I fine tuned the scale and search area for each scale plus the step size to optimize my model. Here are some example images:

![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames and their corresponding heatmaps:

![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

When I used all three features, including spatial color, color hist and HOG, I get many false positive detection for the test video. Then I stopped using spatical color and color hist features, but just HOG feature. I start off model with default parameters as we used during class, the model accuray is ok (~97%), and the false postive issue is improved, but still not good enought. Then I exclude the training data from folder `GTI_Left` and `GTI_Far`, because we dont have similar view angle vehicles as these 2 folder presents.

To save time, I start off 16 as valude for `pixels_per_cell` and 11 as `orentation`, but it turns out not good for detect vehicles in test and project videos, beacuse the extract HOG is too course, the circle like gradient feature is not obvious for the vehicles. I ended up using 10 for
`pixels_per_cell`.

For the sliding window, I implement multiple scale size to detect the vehicles, and it turned out the size of 1~2 is good for the videos in this project, and the size greater than 2 is not able to detect vehicle well. So the I implement 1,1.5 and 2 as my scale size. The smaller scale is used on the top portion of videos, the larger scale is employed at the bottom part of videos, because the size of vehicle is smaller at the far distance.

I used dynamic threshold values for different frame. The positive detection of 10 frames are added together to eleminate false positive detection. However, we dont have the detection of 10 frames at the beginning of the videos, therefore I do see the failure as shown in the following frame, which is the very beginning frame of the test video.

![alt text][image20]

The final pipeline still has few false detection as shown in the following 2 detections. The possible mitigations to make the algorithm robustic and solve this issue are:
 1) improving the training accuracy by using more relavent data, so the false positive detections can be eleminated
 2) Use more layers of ML model, such as neural network to improve the accuray 
 3) Increasing the threshold value of the `apply_threshold` function
 4) Implement DNN

![alt text][image18]
![alt text][image19]

