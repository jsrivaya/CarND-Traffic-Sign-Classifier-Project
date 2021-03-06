#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/visualization.png "Visualization"
[image2]: ./writeup/dataAugmentation1.png "Preprocessing methods example"
[image3]: ./writeup/dataAugmentation2.png "Preprocessing methods example"
[image4]: ./writeup/testImages.png "Set of Traffic Signs"
[image5]: ./writeup/expectedImages.png "Expected from the Training Set"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

Here is a link to my [project code](https://github.com/jsrivaya/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

I used python and numpy to calculate statistics on the dataset:

* The size of training set is ? 34799
* The size of the validation set is ? 4410
* The size of test set is ? 12630
* The shape of a traffic sign image is ? 32x32 RGB
* The number of unique classes/labels in the data set is ? 43


####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I have included a table generated in python where we can see the percentage of images for each class/label

Type Image | Total Number | % in Training Set    | Sign Name
---------- | ------------ | -----------------    | ---------
0	   | 180	  | 0.517		 | Speed limit (20km/h)
1	   | 1980	  | 5.690		 | Speed limit (30km/h)
2	   | 2010	  | 5.776		 | Speed limit (50km/h)
3	   | 1260	  | 3.621		 | Speed limit (60km/h)
4	   | 1770	  | 5.086		 | Speed limit (70km/h)
5	   | 1650	  | 4.742		 | Speed limit (80km/h)
6	   | 360	  | 1.035		 | End of speed limit (80km/h)
7	   | 1290	  | 3.707		 | Speed limit (100km/h)
8	   | 1260	  | 3.621		 | Speed limit (120km/h)
9	   | 1320	  | 3.793		 | No passing
10	   | 1800	  | 5.173		 | No passing for vehicles over 3.5 metric tons
11	   | 1170	  | 3.362		 | Right-of-way at the next intersection
12	   | 1890	  | 5.431		 | Priority road
13	   | 1920	  | 5.517		 | Yield
14	   | 690	  | 1.983		 | Stop
15	   | 540	  | 1.552		 | No vehicles
16	   | 360	  | 1.035		 | Vehicles over 3.5 metric tons prohibited
17	   | 990	  | 2.845		 | No entry
18	   | 1080	  | 3.104		 | General caution
19	   | 180	  | 0.517		 | Dangerous curve to the left
20	   | 300	  | 0.862		 | Dangerous curve to the right
21	   | 270	  | 0.776		 | Double curve
22	   | 330	  | 0.948		 | Bumpy road
23	   | 450	  | 1.293		 | Slippery road
24	   | 240	  | 0.690		 | Road narrows on the right
25	   | 1350	  | 3.879		 | Road work
26	   | 540	  | 1.552		 | Traffic signals
27	   | 210	  | 0.603		 | Pedestrians
28	   | 480	  | 1.379		 | Children crossing
29	   | 240	  | 0.690		 | Bicycles crossing
30	   | 390	  | 1.121		 | Beware of ice/snow
31	   | 690	  | 1.983		 | Wild animals crossing
32	   | 210	  | 0.603		 | End of all speed and passing limits
33	   | 599	  | 1.721		 | Turn right ahead
34	   | 360	  | 1.035		 | Turn left ahead
35	   | 1080	  | 3.104		 | Ahead only
36	   | 330	  | 0.948		 | Go straight or right
37	   | 180	  | 0.517		 | Go straight or left
38	   | 1860	  | 5.345		 | Keep right
39	   | 270	  | 0.776		 | Keep left
40	   | 300	  | 0.862		 | Roundabout mandatory
41	   | 210	  | 0.603		 | End of no passing
42	   | 210	  | 0.603		 | End of no passing by vehicles over 3.5 metric tons
---------- | ------------ | -----------------    | -----------------
Total = 34799
Total % = 100.000

![alt text][image1]

###Design and Test a Model Architecture

I created a set of help functions to support the preprocessing and the trainning set augmentation.
I followed [Pierre Sermanet and Yann LeCun](yann.lecun.org/exdb/publis/psgz/sermanet-ijcnn-11.ps.gz) paper on traffic sign recognition.
For the preprocessing I do convert the image into YUV color space, apply equalize hist on channel Y, apply an INTER_LANCZOS4 normalization filter and convert it back to RGB. The image is also scaled down to 32x32 in case it's not. The INTER_LANCZOS4 normalization uses up to 8 pixels around to normalize the value of each pixel. I did try other normalization techniques but this one looks like providing the better results.

One step that produces significant improvements for validation accuracy is the data set augmentation. As mentioned before I created help methods to do this. Amon those are, bluring the image, rotating left and right on it's center, rotating the image using an image corner, displacing the image, scaling the image down and refill it, converting it to YUP color space and converting it to grayscale. The dataset was augmented from 34799 images to 382789 images. This can cause overfitting but we will show how we prevent that to happen.

Technically training the CNN using grayscale should better results. The reason for this is that the CNN would have less channels to learn and would be able to generalize better. In my case I did try it but I wasn't able to get significant improvements. Other areas might have need improvements at that moment.

Bellow there is a couple of examples of how the support methods for data augmentation work. As well as how the preprocess method transform an original image. For Image labeled 9 and 10 there is a significant improvement.

![alt text][image2]

![alt text][image3]

My final model consisted of the following layers:

| Layer         	      	|     Description	        				                 	| 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x3 RGB image   							                   | 
| Convolution 3x3     	 | 1x1 stride, same padding, outputs 32x32x64 	  |
| RELU					             | Activation	                                 		|
| Max pooling	      	   | Input = 28x28x6, 2x2 stride, Output = 14x14x6 |
| Convolution 3x3	      | Output = 10x10x16.                   									|
| RELU					             | Activation	                                 		|
| Max pooling	      	   | Input = 10x10x16, 2x2 stride, Output = 5x5x16	|
| Fully connected		     |                                      									|
| Dropout        		     | 80% chances of retention        								     	|
| Fully connected		     |                                      									|
| Dropout        		     | 80% chances of retention             									|
| Fully connected		     |                                      									|


I did follow and interactive process to tune the model

* The model was trained using a batch size of 128, and a total of 50 EPOCHS.
  I did try low number of epocs at the begining but realized that it would get better results with a larger number of EPOCHS. Increasing the EPOCHS to more than 50 would also produce and overfiting making the model stop learning and start producing worst results. 50 looked like the sweet point.
* I kept using the AdamOptmizer although I did consider exploring others. That was definitly an area I should have explored more.
* Reducing the learning rate to 0.001 does produce significant improvements in the model. This help the CNN to get out of local minimums and progress toward better errors.
* Dropout Layers. I included two dropout layers between the flat layers. This technique was demostrated as very efficient. Specially with the large data set. I did try different dropout probabilities and although 0.5 and 0.5 did work at the begining it would drop too many features for the model to learn.
* Regularization. I did follow [Cristina Escheau](https://chatbotslife.com/regularization-in-deep-learning-f649a45d6e0) post on Regularization for applying this technique. This was also demostrated very efficient to reduce the error and balance the weights. I did apply regularization only for the weights of the convolutional layers. I did follow a different technique previous to applying regularization that allowed me to get faster and actually better results. I reduced the standard deviation to 0.005 for the weights generation.
* Biases of ones instead of zeros. This initialization accelerates the early stages of learning by providing the ReLUs with positive inputs. I did follow [Alex Krizhevsky Ilya Sutskever Geoffrey E. Hinton](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) paper.

My final model results were:
* training set accuracy of ? 95.8%
* validation set accuracy of ? 95.8%
* test set accuracy of ? 92.5%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? I did try the original LeNet5 architecture. I did choose this because it looks like a basic simple model to get a baseline. I got up to 87% accuracy with default parameters.
* What were some problems with the initial architecture? The initial architecture show issues as soon as I bumped up the training set. The learning rate was too high and the standard deviation for the weights was not helping either.
* How was the architecture adjusted and why was it adjusted? I did include a couple of dropout layers. As mentioned before it did help to strenght the generalization of the model.
* Which parameters were tuned? How were they adjusted and why? As mentioned before I used regularization to minimize the error, adding this to the optimizer. I did use regularization for the convolutional layers only. The learning rate was also reduce and the number of epocs increased.
* What are some of the important design choices and why were they chosen? I did explain already the important and positive effects that Dropout and Regularization had in the model.

###Test a Model on New Images

The model was able to recognize 4/6 images.

* The first image might be difficult to classify because the training set image is a really hard one to train on. Nevertheless the model was able to identify the image as the second most probable image.
* The second image is an image that is not in the set but that it is similar to one that is in the set, narrow on the right. The model was able to identify it as the most similar one.
* The next images 3 images should not be any hard to identify and it is not, the accuracy is impresively high, ~99.9%.
* The last image included in the set is a highly covered in snow image. The model is capable of recognizing the image with a 61% accuracy.

Here are five German traffic signs. Original image, preprocessed image and expected image based on the validation set.

![alt text][image4]
![alt text][image5]

Here are the results of the prediction:

| Image			        |     Prediction	        					| Certinity
|:---------------------:|:-----------------------------------------------:| :-------
| No passing      	          	| No passing for vehicules over 3.5 tones | 51%
| Narrow at both sides     			| Narrow on right 									      | 99.9%
| Wild Animal Crossing					| Wild Animal Crossing									  | 99.9%
| Pedestrians	      	         	| Pedestrians					 			              | 99.9%
| Right-of-Way 			            | Right-of-Way      							        | 91.1%
| General Caution 			        | General Caution      							      | 61.6%


The code for making predictions on my final model is located in the 2nd to last cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a No passing sign (probability of 0.51). It is in fact a no passing sign, but not exactly the one that it predicted. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .51         			    | No passing for vehicules over 3.5 tones | 
| .20     				      | No passing 										|
| .15					          | Speed limit 50kmh										|
| .02	      		      	| Turn right ahead					 				|
| .01				            | Roundabout mandatory     							|


For the second image, the model is very sure that this is a narrow on the right. But this is an image that does not exist in the training. It is a narrow on both sides sign. All the other predictions are extremely close to zero.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			    | Narrow on right | 
| .0     			  	      | Pedestrians 										|
| .0					          | Right-of-way next intersection										|
| .0	      		      	| Be aware of ice/snow					 				|
| .0				            | Dangerous curve to the left     							|

For the third image, the model is very sure that this is a 'Wild Animal Crossing' sign. And it is correct. All the other predictions are extremely close to zero.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			    | Wild Animal Crossing | 
| .00     				      | Double curve 										|
| .00					          | Slipery road										|
| .00	      		      	| Be aware of ice/snow					 				|
| .00				            | Bycicle crossing     							|

For the fourth image, same as with previous image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			    | Pedestrians | 
| .00     				      | Narrow on the right 										|
| .00					          | General Caution										|
| .00	      		      	| Right of way next intersection					 				|
| .00				            | Traffic signal     							|

For the fifth image, same as with previous image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .91         			    | Right-of-Way | 
| .04     				      | Pedestrians 										|
| .02					          | Road narrows on the right										|
| .02	      		      	| Be aware of ice/snow					 				|
| .00				            | Dangerous curv to the right     							|

For the sixth image, the model is capable of recognizing the image despite the fact that the sign is almost covered by the snow. The model is relatively sure that this is the sign. And it is right

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .61         			    | General Caution | 
| .38     				      | Childre Crossing 										|
| .00					          | Bicycle Crossing										|
| .00	      		      	| No vehicles					 				|
| .00				            | Bumpy Road     							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


