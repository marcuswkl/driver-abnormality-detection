# driver-abnormality-detection

### Abnormality Detection in Driver Behaviour on City Roads (Simulation for Data Collection)
**Final Year Project for BSc (Hons) in Computer Science**

The high rate of road accidents in Malaysia has led to many fatal cases which can be attributed to irresponsible, negligent drivers with abnormal driver behaviour. This behaviour should be detected so that corrective measures can be taken. However, it is difficult to identify abnormalities in driver behaviour through traditional programming methods. Hence, a machine learning approach is proposed for abnormality detection in driver behaviour on a city roads environment.

## Data
- Normal and abnormal driver behaviour data collected from participants using a driving simulator and [CARLA](https://github.com/carla-simulator/carla).
- Trimmed, vectorised and combined to create a master dataset.
- Analysed and pre-processed using feature selection, engineering, aggregation and scaling.

## Models
- Classification
  - Support vector machine (SVM)
  - Random forest
  - Artificial neural networks (ANN)

## Evaluation Metrics
- Confusion matrices
- Accuracy, precision and recall
- F1-score

## Tech Stack
- Python
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
