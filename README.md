# Prediction_of_Patients_Deterioration_in_The_Cardiac_Wards

This project focuses on detecting deterioration of acutely ill patients in the cardiac ward at the University of Virginia Health System. Patients in the cardiac ward are expected to recover from a variety of cardiovascular procedures, but roughly 5% of patients deteriorate and have to be transferred to the Intensive Care Unit (ICU) for elevated care. This is an important problem because the probability of mortality increases by hour for the patients that are delayed to get into the ICU. In this work, a super learner was built by stacking logistic regression, random forest, and gradient boosting models. Given that only 1% of observations are labeled as events, the F1 score was used as the primary metric to assess the performance of each model; area under the curve (AUC) was also considered. 

## Data 

**Dataset**

The data were collected from 71 monitored beds in three cardiac-related wards at the University of Virginia Health (UVA) System from Oct 11th, 2013 to Sept 1st, 2015. It contains 63 patient-years of data, including vital signs, lab results and ECG monitoring data from 8,105 acute care patient admissions. All monitored beds had the ability to collect and store continuous ECG data from the patients. For each patient, vital signs were recorded by a nurse every four hours, lab results were taken every 12 hours, and the continuous ECG-derived data were calculated every 15 minutes. This information was then combined with patient outcomes and general demographic information. 
_The dataset is public in UVA DataVerse_.

**Labelling**

The outcomes were represented by binary values, with “1” indicating an event and “0” indicating a non-event. Event means transferring to ICU or death. As shown in Figure I, if the patient never had to be transferred to ICU during his/her stay, his/her records were all labeled as “0”; Otherwise, the record started being labeled as “1” 24 hours prior to the transfer. 

<img width="340" alt="event_definition" src="https://user-images.githubusercontent.com/42804316/57731585-fcb66900-7668-11e9-9b40-b2a8620ab3b0.png"><br />*Recorded every 15 mins

**Data Preprocessing**

All missing lab and vitals values were imputed via sample and hold from observations up to 48 hours prior through the following 15 minutes chunks until next step updated. The remaining missing observations (less than 1%) were imputed with the median values. The dataset was highly imbalanced with only 1% of entries classified as events out of two million records. Synthetic Minority Over-Sampling Technique (SMOTE) is implemented to increase the number of the minority class instances and downsample the majority class as well. Though the dataset contained repeated measures, each record was treated as an independent observation.

**Autoencoder**

The autoencoder neural network uses the input features as the targets and applys back propagation to optimize the weights to obtain the machine-generated features. It has two encoder layers and two decoder layers. Gaussian noise was incorporated into the input data and then the noisy data was mapped to clean data to enhance its generalization. Additionally, two regularization techniques, dropout and L1 regularization, were employed to decrease the the likelihood of overfitting. By using the denoising and sparse autoencoder model, compressed features were obtained and then further fitted to the super learner model. In our case, the encoder included an input layer of 49 features, a second layer of 30 features and a bottleneck with 15 nodes, corresponding to the 15 abstracted features calculated by the network. 

<img width="420" alt="autoencoder" src="https://user-images.githubusercontent.com/42804316/57733465-94b65180-766d-11e9-83b5-0884e4bced92.png">

## Metrics Used

Accuracy was not appropriate here due to the class imbalance. Although AUC showcases model performance across a wide range of thresholds and risk levels, a metric that better expresses model performance at low false positive rates was required to avoid alarm fatigue. Therefore, the bottom left hand corner of a Receiver Operating Characteristic (ROC) curve (partial ROC curve) was focused because it corresponds to the lower false positive rate.

Another metric used was the F1 score, which is the harmonic mean of precision and recall. A good F1 means that the model identifies the real event and avoids being disturbed by false alarms. The F1 score yields a more accurate reflection of model performance at low false positive rates. 

## Machine Learning Algorithms

**Logistic Regression** **Random Forest** **Extreme Gradient Boosting (XGBoost)**

Logistic regression is robust to class imbalance, but lacks in flexibility. The risk of clinical deterioration is not an additive function of patients’ physiological parameters. Crossing certain thresholds of some variables are more likely to be an indication of high risk which might be missed in logistic regression. Comparatively, tree-based models are more appropriate for our application because they were designed to find these thresholds. It adds additional randomness to the training process which provides a more accurate and stable prediction. XGBoost is another popular supervised learning algorithm for solving classification problems, reducing running time by implementing parallel processing.

**Super Learner**

Considering different models may detect different patients, so a super learner was developed by stacking different models together. The idea behind this approach was that taking all of the individual model predictions into account and combining them into a single prediction would be more effective in detecting a wider variety of patients than any single model. The super learner was fed predicted probabilities from the candidate learners (random forest, logistic regression and XGBoost) and combined them into a single probability. Rather than build a single model using all of the features, a probability was generated per candidate learner per category of feature. These probabilities were, in turn, used as inputs to a larger logistic regression model to produce a probability for each observation in the dataset. The architecture of super learner is shown as below.

<img width="600" alt="super_learner" src="https://user-images.githubusercontent.com/42804316/57736183-16f64400-7675-11e9-9c1b-dbdff893e665.png">

## Results

Random forest has the highest AUC and exceeds super learner by a very little. But based on F1, the super learner leads among all the models. 
As mentioned, partial ROC curves was focused which corresponds to the high thresholds and helps prevent the alarm fatigue for clinicians. In the partial ROC curves, the super learner separates itself from other models, which are reflected well by F-1 score. It shows that the F1 is the desirable metric to evaluate model’s performances in applications with low event rates.

<img width="50%" alt="bar_plot" src="https://user-images.githubusercontent.com/42804316/57736749-21b1d880-7677-11e9-9b20-0544dd5eaf98.png"><img width="50%" alt="ROC_curve" src="https://user-images.githubusercontent.com/42804316/57736862-8ff69b00-7677-11e9-9e5f-c01d2a0df589.png">

In conclusion, the random forest and super learning perform the best in our case, and F1 score can best represent the model’s performance in the application with low event rate.
















