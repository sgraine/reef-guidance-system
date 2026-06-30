import seaborn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import rc

def calculate_f1_scores(conf_matrix):
    """
    Calculate per-class and overall F1 scores from a confusion matrix.
    
    Args:
        conf_matrix (numpy.ndarray): Square confusion matrix (C x C) where C is the number of classes.
    
    Returns:
        dict: Per-class F1 scores and overall F1 score.
    """
    # Extract true positives, false positives, and false negatives
    tp = np.diag(conf_matrix)
    fp = np.sum(conf_matrix, axis=0) - tp
    fn = np.sum(conf_matrix, axis=1) - tp
    
    # Calculate precision and recall
    precision = np.where((tp + fp) > 0, tp / (tp + fp), 0)
    recall = np.where((tp + fn) > 0, tp / (tp + fn), 0)
    
    # Compute per-class F1 scores
    f1_scores = np.where((precision + recall) > 0, 2 * (precision * recall) / (precision + recall), 0)
    
    # Compute overall F1 score (macro-F1)
    overall_f1 = np.mean(f1_scores)
    
    return {"per_class_f1": f1_scores, "overall_f1": overall_f1}

if __name__ == "__main__":

    # conf_matrix = np.array([[2325,539],
                            # [389,750]])

    # conf_matrix = np.array([[2207,657],
    #                         [343,796]])

    # conf_matrix = np.array([[2031,833],
    #                         [279,860]])

    # f1_results = calculate_f1_scores(conf_matrix)
    # print("Per-class F1 scores:", f1_results["per_class_f1"])
    # print("Overall F1 score:", f1_results["overall_f1"])

    # rc('text', usetex=True)
    
    seaborn.set(font_scale = 1.3)

    # # ######################################### THRESHOLD TRIALS: F1 SCORE ##########################################################
    # point_labels_10 = np.array([[0.7257122],[0.7147456],[0.69621193]]) # F1 0.6, 0.5, 0.4
    # point_labels_10 = np.array([[0.47405],[0.561229],[0.590031],[0.6199],[0.6025]]) # f1 0.6, 0.5, 0.4
    point_labels_10 = np.array([[0.4245],[0.5003],[0.5548],[0.5938],[0.6160],[0.6325]]) # f1 0.6, 0.5, 0.4

    # point_labels_10 = np.array([[0.8567],[0.8655],[0.8792]]) # prec 0.6, 0.5, 0.4
    # point_labels_10 = np.array([[0.8118],[0.7706],[0.7091]]) # rec 0.6, 0.5, 0.4

    fig, ax = plt.subplots(ncols=1, sharey=True, figsize=(3,4))
    sns_g = seaborn.heatmap(point_labels_10*100, ax=ax, cbar=False, square=True, cmap=mpl.colormaps['YlGn'], annot=True, fmt='.2f')

    sns_g.tick_params(axis="y", rotation=0)

    ytick_positions = np.arange(0.5, 6.5, 1)
    ytick_labels = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

    sns_g.set_yticks(ytick_positions)
    sns_g.set_yticklabels(ytick_labels)
    sns_g.tick_params(left=False)

    fig.text(0.04, 0.5, r'$\alpha$', va='center')
    plt.show()
    plt.savefig("Heatmap-Threshold-f1-v2.pdf", bbox_inches='tight')

    # ########################################## HEATMAP: CONFUSION MATRIX ##########################################################

    # Example confusion matrix (true labels are rows, predicted are columns)
    # conf_matrix_percent = np.array([
    #         [77.58, 22.42],
    #         [29.76, 70.24]
    # ])
    conf_matrix_percent = np.array([
            [82.00, 18.00],
            [32.84, 67.16]
    ])

    # Class names
    class_names = ['No Deploy', 'Deploy']

    # Create the plot
    fig, ax = plt.subplots(figsize=(4, 4))
    sns_g = seaborn.heatmap(
        conf_matrix_percent,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 18},  # font size for numbers
        cmap=mpl.colormaps['Purples'],
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar=False,
        linecolor='black',
        linewidths=1,
        ax=ax
    )

    # Move x-axis labels to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # Set axis labels with bold font and bigger size
    ax.set_xlabel(r'$\mathbf{Predicted\ Label}$', labelpad=10, fontsize=24)
    ax.set_ylabel(r'$\mathbf{True\ Label}$', fontsize=24)

    # Rotate tick labels and increase font size
    sns_g.set_xticklabels(class_names, rotation=45, ha='left', fontsize=18)
    sns_g.set_yticklabels(class_names, rotation=0, fontsize=18)

    plt.tight_layout()
    plt.show()
    plt.savefig("Confusion-Matrix-Heatmap-TopLabels-v2.pdf", bbox_inches='tight')