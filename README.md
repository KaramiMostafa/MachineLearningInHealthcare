# MachineLearningInHealthcare
Part I : **Melanoma Detection (Skin Cancer)**

 Among the different types of skin cancer, melanoma is considered to be the deadliest and is difficult to treat at advanced stages. Detection of melanoma at earlier stages can lead to reduced mortality rates. Melanoma often looks like a mole. It is usually black or brown, but it can also be skin-colored, purple, blue, red, pink, or even white. Melanoma represents one of the most common cancer in the world, and one of the specific strategies for early recognition of the disease is the ABCDEs technique.

* ***A*** is for ***Asymmetry***: Most melanomas are asymmetrical. If you draw a line through the middle of the lesion, the two halves do not match, so it looks different from a round to oval and symmetrical common mole
* ***B*** is for ***Border***: Borders tend to be uneven and may have scalloped or notched edges, while common moles tend to have smoother evener borders.
* ***C*** is for ***Color***: Multiple colors are a warning sign.
* ***D*** is for ***Diameter***: Benign moles usually have a smaller diameter than malignant ones.
* ***E*** is for ***Evolution***: Any change in size, shape, color or elevation of a spot on the skin, or any new symptom in it, such as bleeding, itching or crusting. 

 To automatically detect skin cancers, teledermatology tools such as nevi picture can be useful to improve the diagnosis quality since, for example, the waiting lists and times are even increasing, and there are too few dermatologists.
 
The available data is presented as set of several jpeg images of moles divided in 3 type:

o (low_risk_n.jpg) for moles that have a low probability of being melanoma
o (medium_risk_n.jpg) for moles that have a low probability of being melanoma
o (melanoma_n.jpg) for moles that have a high probability of being melanoma

The purpose is to find and extract the correct feature that involves the border and asymmetry aspect. The K-means algorithm is used to detect the presence of a mole in the image. The images are split into K different clusters of pixels based on the pixel’s color. A value of K=3 is enough in most of the images to detect the mole, while on some pictures, K=4 (medium_risk_24) and K=5 (melanoma_27) were used due to the slight difference between the color of skin and mole. Our focus is on the cluster, that groups the darkest pixels, and by evaluating its contour, it is possible to calculate the perimeter and area of the mole, which leads to extract border and asymmetry as features.
Some color preprocessing steps are needed before extracting the features, which include quantization of the images, preparing the background, extract the binary image of the moles, getting the digitized picture of the moles, and improving the digitized image.


Part II : **Regression on Parkinson Data**

 Parkinson’s disease (PD) is a neurodegenerative condition that affects nerve cells in the brain that control movement. Parkinson’s is progressive, which means symptoms appear gradually and slowly get worse. Everyone with Parkinson’s has different symptoms, but the most common sign is muscle rigidity and slowness of movement, which depends on the severity of the illness.
 
 By evaluating the body movements and voice of patients, neurologists can obtain UPDRS (Unified Parkinson’s Disease Rating Scales), which is the final grade of illness. Since visiting patients takes a lot of time and different neurologists may assign different scores, it would be much more efficient to create a way for checking patients in a way that the practical difficulties could be overcome by automatizing the evaluation of the UPDRS. It is possible to use features (as our parameters) like voice to predict the total UPDRS. It is essential to mention that Parkinson’s disease does not always affect voice, which leads us to believe the mentioned claim may not be entirely correct.

 Using the regression algorithm helps to find out if there is any correlation between features and total UPDRS or not, and also which one is the best. Regression algorithms are going to used values of the other features to predict total UPDRS. Also, it is essential to see the comparison between estimation and the true value of total UPDRS, which helps to evaluate the accuracy of regression. The goal is to use the dataset to estimate the total UPDRS.

 The dataset is composed of several biomedical voice measurements from 42 people with early-stage Parkinson’s disease. Rows represent the voice records, while each column refers to a particular feature such as subject number, subject age, subject gender, the time interval from baseline recruitment date, motor UPDRS, and total UPDRS.
The Matrix containing all the data has been divided into three sub-matrices:
- Ⅰ. 50% of the data used as training data (examples used for learning)
- Ⅱ. 25% of the data as validation data (to know how well our model has been trained)
- Ⅲ. 25% of the data as test data (used to evaluate the final model performance)

 The data should be normalized and shuffled since it reduces variance and makes sure that the models remain general and overfit less. Moreover, normalization makes features comparable. It is essential to mention that by using the mean and standard deviation of the training dataset, normalizing all three parts of the whole dataset is possible due to the fact that part of the information is in feature and by using only training data for normalization, it is possible to test and evaluate whether the model can generalize well to new, unseen data points. Data standardization procedures generally equalize the range and/or data variability.
