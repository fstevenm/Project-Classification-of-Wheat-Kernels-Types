## PROJECT ARTIFICIAL NEURAL NETWORKS - MACHINE LEARNING

#### Project-Classification-of-Wheat-Kernels-Types-Using-Backpropagation

The aim of prepare this paper is to provide classification methods that have ability to distinguish seed types named Canadian, Rose, and Kama from each other according to their geometrical features. Kernels datasets picked up from UCI machine learning database.

Here is the link dataset:

    https://archive.ics.uci.edu/ml/datasets/seeds

The number of geometrical features is seven (area , perimeter , compactness, length of kernel, width of kernel, asymmetry coefficient , length of kernel groove). Wheat is classified by applying backpropagation method in the Python program. Backpropagation is one of methods of artificial neural networks that can be used in solving problems identification or classification.


Each variety was taken as many as 70 for the experiment. Here's a diagram bar of the number of each label (class):
![image](https://user-images.githubusercontent.com/99526319/162606766-671b359d-b41f-4f83-9e03-cb66afcd11f3.png)


From the data sources used, there are 3 labels (classes) or 3 categories, namely:
1. Label 1 (Showing Kama variety wheat)
2. Label 2 (Showing Rosa variety wheat)
3. Label 3 (Shows Canadian varietal wheat)

Steps in
Backpropagation methods are:
1. Forward input
2. Calculating error
3. Backpropagate error
4. Updating weights and bias

The activation function used is the sigmoid function. The sigmoid function has the formula that is

#### Network architecture
![image](https://user-images.githubusercontent.com/99526319/162606846-7ef30107-4051-41f2-91e6-c7110837b871.png)


### Prediction data visualization

#### The graph plot of the relationship between the number of neurons in the hidden layer and accuracy
![image](https://user-images.githubusercontent.com/99526319/162606883-d374eef2-1ce5-4f7d-b9cd-03ac852b2063.png)

#### The confusion matrix
![image](https://user-images.githubusercontent.com/99526319/162606976-f27791b5-00cf-4a73-9321-4356cf035b9f.png)

![image](https://user-images.githubusercontent.com/99526319/162607044-a38b9092-9203-46be-931a-1d4e963827e6.png)

Based on the confusion matrix, from 70 testing data, there are 27 seed varieties data Kama wheat, 19 data on Rosa wheat seed varieties, and 24 data on wheat seed varieties Canadian. It can be seen that there are 24 test data that are correctly predicted as varieties Kama (34.29% of the testing data), while the other 3 Kama varieties testing data incorrectly predicted as Canadian (4.29% of testing data). Meanwhile, 19 testing data (27.14% of the test data) was successfully predicted correctly as a wheat seed variety Rosa and 24 testing data (34.29% of the testing data) were predicted correctly as a Canadian wheat seed variety.

#### Result Recap
![image](https://user-images.githubusercontent.com/99526319/162607015-b26e9e6a-6b77-4957-ac88-6eac1d624a04.png)

#### Limitation
The limitation in this study is the lack of available data from sources data even though the accuracy is quite good. If the amount of data is more, it is possible that the accuracy results to be obtained will be better (increase).

#### Conclusion
Based on the research that has been done, the Backpropagation method is able to create an artificial neural network system that is useful in the classification of seed varieties wheat with good accuracy, which is 95.71% with the number of neurons in the hidden layer as many as 10 (can be seen in the table of analysis results).

#### Suggestion
Readers can apply this method to classify seed varieties wheat because it has a good level of accuracy. It's better to do it this way or another method to find out if there is another method that produces accuracy the best.

You can see this full project in the folder 'Project Classification of Wheat Kernels Types - Backpropagation [Artificial Neural Networks]'

#### Don't forget to provide the source if you use this project!
