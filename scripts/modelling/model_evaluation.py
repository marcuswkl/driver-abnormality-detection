import matplotlib.pyplot as plt
import numpy as np

# Visualise the comparison of model scores using grouped bar chart
def compare_model_scores(model_label_names, metric_names, scores_list):
    x = np.arange(len(metric_names))
    width = 0.3
    x_positions = [x-width, x, x+width]
    
    # Plot bar chart for each model
    fig, ax = plt.subplots()
    for index in range(0, len(model_label_names)):
        plt.bar(x_positions[index], scores_list[index], width, label=model_label_names[index])
    
    plt.title("Comparison of Model Performance Metric Scores")
    plt.xticks(x, metric_names)
    plt.xlabel("Performance Metrics")
    plt.ylabel("Scores")
    plt.legend()
        
    fig.tight_layout()
    plt.show()