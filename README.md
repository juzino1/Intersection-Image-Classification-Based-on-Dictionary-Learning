# Intersection-Image-Classification-Based-on-Dictionary-Learning
Pattern Recognition，SVM，intersection，Multi-classification
One of the most obvious features implemented in this project is multi-classification.The dataset is for intersection classification.
Algorithm structure
1. Prepare data: divide the training data (known categories) and test data (to be classified).
2. Extract features: find the "marker points" in each training image and describe their features with numbers.
3. Generate a dictionary: cluster the features of the mark points of all training images to form a set of "visual word lists".
4. Construct feature vectors: use the dictionary to count the frequency of "visual words" in each image and obtain a histogram (feature vector) of fixed length.
5. Train classifier: use the histogram and category information of the training image to train a support vector machine classifier.
6. Test classification: repeatedly extract features and count histograms for the test image, and then use the classifier to predict the image category
   Very easy to reproduce
