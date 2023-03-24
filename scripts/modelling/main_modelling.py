# DATA UNDERSTANDING

import pandas as pd
import numpy as np

df = pd.read_csv('dataset.csv')

# Examine the dataset
print(df.describe().transpose())
print()

# Visualise distribution of target values
import matplotlib.pyplot as plt

plt.figure()
plt.title("Target Distribution")
plt.xlabel("Abnormality Classes")
plt.xticks(np.arange(2), ['Normal', 'Abnormal'])
plt.ylabel("Count")
plt.hist(df['abnormality'], bins=3)

# DATA PREPROCESSING

# Remove unused features
df_removed = df.drop(columns=['time', 'yaw', 'heading', 'location_x', 'location_y', 'gnss_latitude', 'gnss_longitude', 'gyroscope_x', 'gyroscope_y', 'height', 'reverse', 'hand_brake', 'manual_gear_shift', 'gear'])

# Engineer new features
import feature_engineering as fteng
df_engineered = fteng.engineer_features(df_removed)

# Aggregate all features
import feature_aggregration as ftagg
df_aggregated = ftagg.aggregate_features(df_engineered)

# Define input features and target
X = df_aggregated[['mean_speed', 'max_speed', 'mean_acceleration', 'max_acceleration', 'mean_yaw_speed', 'max_yaw_speed', 'mean_throttle', 'max_throttle_count', 'mean_steer_change', 'max_steer_change', 'mean_positive_brake_change', 'max_positive_brake_change']]
Y = df_aggregated['abnormality']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7)

# Scale input features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# MODEL TRAINING

# Construct the support vector machine (SVM) model using support vector classifier (SVC)
from sklearn.svm import SVC
svc = SVC()

# Construct the random forest model using random forest classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

# Construct the artificial neural network (ANN) model using multilayer perceptron (MLP) classifier
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(7), max_iter=2000)

# Train the models
svc.fit(X_train, y_train)
rfc.fit(X_train, y_train)
mlp.fit(X_train, y_train)

# MODEL VISUALISATION

# Define names for visualisation
feature_names = ['mean_speed', 'max_speed', 'mean_acceleration', 'max_acceleration', 'mean_yaw_speed', 'max_yaw_speed', 'mean_throttle', 'max_throttle_count', 'mean_steer_change', 'max_steer_change', 'mean_positive_brake_change', 'max_positive_brake_change']
target_names = ['normal', 'abnormal']

import model_visualisation as mdvis

# No visualisation for 12 dimensional support vector machine (SVM) model

# Visualise a random decision tree in the random forest model
mdvis.visualise_dt(rfc, feature_names, target_names)

# Visualise the artificial neural network (ANN) model
mdvis.visualise_ann(mlp)

# MODEL EVALUATION

# Predict the target values for testing dataset using each model and save the results into a list
y_test_pred_list = [svc.predict(X_test), rfc.predict(X_test), mlp.predict(X_test)]

# Evaluate the models using confusion matrices
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

for y_test_pred in y_test_pred_list:
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred)).plot()

# Evaluate the models using performance metrics
from sklearn.metrics import classification_report

model_names = ["Support Vector Machine (SVM)", "Random Forest", "Artificial Neural Networks (ANN)"]
for index, y_test_pred in enumerate(y_test_pred_list):
    print(model_names[index])
    print(classification_report(y_test, y_test_pred, target_names=target_names, digits=4))

# MODEL COMPARISON
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import model_evaluation as mdeval

# Obtain the performance metric scores of each model
svc_scores = []
rfc_scores = []
mlp_scores = []
scores_list = [svc_scores, rfc_scores, mlp_scores]

for index, y_test_pred in enumerate(y_test_pred_list):
    scores_list[index].append(accuracy_score(y_test, y_test_pred))
    scores_list[index].append(precision_score(y_test, y_test_pred, average='macro'))
    scores_list[index].append(recall_score(y_test, y_test_pred, average='macro'))
    scores_list[index].append(f1_score(y_test, y_test_pred, average='macro'))

# Compare the performance metric scores of the models using grouped bar chart
model_label_names = ["SVM", "RF", "ANN"]
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
mdeval.compare_model_scores(model_label_names, metric_names, scores_list)