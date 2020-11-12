# Human Activity Recoginition Bagging SVC, RF, GNB

# Introduction

This document details the methodology and procedures carried out in Lab 2 corresponding to Human Activity Recognition. For this, we start from a data set divided into training and test. The data correspond to labelled sequences corresponding to eight people for the training part and two for the testing part. These people carry out a series of five activities that is the data to predict. The five features used are:

  1. Axis Z transformed acelerometer data
  2. Axis XY transformed acelerometer data
  3. Axis X transformed gyroscope data
  4. Axis Y transformed gyroscope data
  5. Axis Z transformed gyroscope data
    
Using these features of the data, a training step will be carried out under three different methods, for then perform the test step and compare the results and execution times. It has been decided to compare three different methods in order to evaluate various methods and procedures to obtain the best results.
The three methods that have been used are: Bagging SVC, Random Forest and Gaussian Naive Bayes. These three methods will be analysed in the document, comparing the results obtained and describing a conclusion with the final method chosen and the best results. It should be noted that Python and the Scitik Learn library were used to execute these methods. Once the best method is selected, the predicted data on the testing part is exported to .mat format as required.

# Prepare the data

The ‘loadmat’ function was used to read the data from the .mat file. Once understood how the data was organized in the data frame and after a visualization of these, we proceeded to go through the obtained data frame that contains all the data to create three: the X training, Y training and X testing.

The X training contains the training data corresponding to the Features, while the Y training contains the training data corresponding to the activity that was being carried out. Finally, in X testing the testing data corresponding to the Features will be stored, from which the activity that was being carried out in each of the observations must be predicted.

Once these data are obtained, we would have an X training of size (141426, 5), in which the rows correspond to the number of singles and the columns to the five features. We also obtain a Y training with size (141426,) which is an array with the same number of samples as X containing the number corresponding to each activity that was being performed for each sample. Finally, X testing would have a shape of (42188, 5). As you can see the number of samples is less since, as previously mentioned, these data correspond to two people, while in training they correspond to eight.

It should be noted that the ‘StandardScaler’ function was used since the methods are influenced by the magnitude of the data, so by standardizing the data of the Features, we procure that they all have the same weight when evaluating the data and are not influenced by the different magnitudes.

# Bagging SVC

Support Vector Classification, this is an SVM applied to classification problems, in this multi-label class. As we know, SVMs use hyperplanes in multidimensional space in order to differentiate between the different classes of the problem. The SVM will try to find the maximum marginal hyperplane that best divides the data set used.
The main problem with this type of method is that it scales quadratically with the number of samples incurring a very long time for problems with a very high number of samples, as is the case. However, SVC provides good results in HAR problems, which is why we looked for ways to speed up the execution of this method.

For this reason, ensembles with bagging (bootstrap aggregating) of SVCs that train in subsets of the dataset were used to reduce the number of samples for each SVC. In other words, each individual SVC is independently trained with a subset of the data set for each of them. Once each SVC is trained, they are aggregated into such that a collective decision is created as the majority voting. Numerous simulations of the literature demonstrate that SVC ensembles with bagging outperforms a single SVM in terms of classification accuracy and time.

To proceed with this topic, the Sklearn method ‘BaggingClassifier’ is used. The n_jobs parameter of these methods was also used so that, when training each SVC, it was carried out in parallel. However, under parallelization, we do not obtain better results. The same happens with the cache_size parameter, trying to give the method more cache, having more memory and resources to run quickly.

It should be noted that in order to extend this method so that it can be used in multi-class classification (as is the case), the “one-against-all” method has been used in which we will have as many SVCs as classes to predict. Although there are also others such as “one against one” in which C (C −1) / 2 classifiers where C is the number of classes would be used. For this, the ‘OneVsRestClassifier’ method from Sklearn is used.

An exhaustive search was used through GridSearchCV with 5 cross fold validation to perform the tuning of the hyper parameters of the method, which in this case are:

* **Kernel:** type of kernel used in the SVC the main function of the kernel is to transform the input data into the required form. Linear was first tried, however rbf offered better results. Rbf is useful for non-linear hyperplanes.
* **Gamma:** kernel coefficient. A low value of gamma considers only nearby points when calculating the separation line, while a high value of Gamma considers all the data points in the calculation of the separation line.
* **C:** regularization parameter used as a penalty which represents the misclassification or error term. This hyper parameter controls the trade-off between boundary decision and misclassification term. A low value of C creates a small-margin hyperplane and a high value creates a larger-margin hyperplane.

# Random Forest

Random forest fits several classifying decision trees on various sub-samples and uses averaging to improve accuracy in predictions and control over-fitting. In this case, the hyper-parameters were also tuned using exhaustive search and 3 cross fold validation. The hyper-parameters that were used in tuning are:

* **n_estimators:** determines the number of trees in the forest.
* **max_features:** maximum number of Features to be considered when looking for the best split.
* **max_depth:** maximum number of tree depth.

# Gaussian Naïve Bayes

Gaussian Naïve Bayes is one of the available Naïve Bayes methods, which are a set of supervised learning algorithms based on applying Bayes theorem with naïve assumptions.
In the case of Gaussian Naive Bayes, the likelihood of the features is assumed to be Gaussian. In this type of method, the parameters used are estimated using maximum likelihood, so it is not necessary to carry out an exhaustive search as in the previous ones. Which will logically reduce the method execution time drastically.

# Final model and conclusions

The following table summarizes the results of the three methods, offering both the execution time of each one and its accuracy:

|                      | Time (seconds) | Score (accuracy) |
|----------------------|----------------|------------------|
| Bagging SVC          | 11146.31       | 94.56 %          |
| Random Forest        | 3698.93        | 97.14 %          |
| Gaussian Naive Bayes | 0.05           | 92.34 %          |

As we can see the method that takes the longest to execute is SVC, which is to be expected because the complexity of the algorithm grows exponentially as the data increases. Therefore, it is an accurate algorithm, but it scales poorly as the size of the data increases.

On the other hand, Random Forest presents a shorter execution time and the highest accuracy of the three. As we can see, the execution time using this algorithm is 66.81% faster, which constitutes a significant performance improvement.

Finally, we find Gaussian Naive Bayes which, despite of being the simplest algorithm of all and not having to perform an exhaustive search in the tuning of its hyper-parameters (since it does not have one), presents the shortest execution time of the three methods, as well as logically the lowest score.

The chosen method to print the results is Random Forest, since it is the best method in terms of accuracy. The data has been exported in .mat format and constitute the prediction of the type of activity performed for each of the two people who make up the test set.
We have also represented the predictions made on the test part for the two people who make up this part, as we can see below:




