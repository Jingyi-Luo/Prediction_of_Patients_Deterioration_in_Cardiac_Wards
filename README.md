# Prediction_of_Patients_Deterioration_in_The_Cardiac_Wards

This project focuses on detecting deterioration of acutely ill patients in the cardiac ward at the University of Virginia Health System. Patients in the cardiac ward are expected to recover from a variety of cardiovascular procedures, but roughly 5% of patients deteriorate and have to be transferred to the Intensive Care Unit (ICU) for elevated care. This is an important problem because the probability of mortality increases by hour for the patients that are delayed to get into the ICU. In this work, a super learner was built by stacking logistic regression, random forest, and gradient boosting models. Furthermore, a denoising auto-encoder was created to generate computer-derived features, the results of which were fed to machine learning models to predict patient deterioration. Given that only 1% of observations are labeled as events, the F1 score was used as the primary metric to assess the performance of each model; area under the curve (AUC) was also considered. 

## Data 

**Dataset**

The data were collected from 71 monitored beds in three cardiac-related wards at the University of Virginia Health (UVA) System from Oct 11th, 2013 to Sept 1st, 2015. It contains 63 patient-years of data, including vital signs, lab results and ECG monitoring data from 8,105 acute care patient admissions. All monitored beds had the ability to collect and store continuous ECG data from the patients. For each patient, vital signs were recorded by a nurse every four hours, lab results were taken every 12 hours, and the continuous ECG-derived data were calculated every 15 minutes. This information was then combined with patient outcomes and general demographic information. 
_The dataset is public in UVA DataVerse_.

**Labelling**

The outcomes were represented by binary values, with “1” indicating an event and “0” indicating a non-event. Event means transferring to ICU or death. As shown in Figure I, if the patient never had to be transferred to ICU during his/her stay, his/her records were all labeled as “0”; Otherwise, the record started being labeled as “1” 24 hours prior to the transfer. 

<img width="419" alt="event_definition" src="https://user-images.githubusercontent.com/42804316/57731585-fcb66900-7668-11e9-9b40-b2a8620ab3b0.png">*Recorded every 15 mins






