""" Confusion matrix visualization
based on https://github.com/DTrimarchi10/confusion_matrix, (C) Dennis Trimarchi

Changes by Martin Drawitsch:
- Support percentage normalization over true or pred conditions
- ax-based instead of figure-based
- Remove group_name support
- Additional spacing between counts and percentages
- Support tick label rotation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          normalize='all',
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          rotatelabels=True,
                          sum_stats=True,
                          cmap='Blues',
                          ax=None,
                          eps=1e-9):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    percent:       If True, show percentages in the confusion matrix. Default is True.

    normalize:     {'true', 'pred', 'all'}, default: 'all'
                       Normalizes confusion matrix over the true (rows), predicted (columns)
                       conditions or all the population.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    rotatelabels:  If True, rotate tick labels outward. Uses more space but fits long label names better.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    '''

    if ax is None:
        fig, ax = plt.subplots()

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    cfsum = float(np.sum(cf) + eps)  # Add 1e-9 against zero-division
    cfsum1 = cf.sum(axis=1, keepdims=True) + eps
    cfsum0 = cf.sum(axis=0, keepdims=True) + eps

    if percent:
        if normalize == 'true':
            group_percentages = ["{0:.2%}".format(value) for value in (cf / cfsum1).flatten()]
        elif normalize == 'pred':
            group_percentages = ["{0:.2%}".format(value) for value in (cf / cfsum0).flatten()]
        elif normalize == 'all':
            group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / cfsum]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}\n{v2}".strip() for v1, v2 in zip(group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / cfsum

        #if it is a binary confusion matrix, show some more stats
        stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
        # if len(cf)==2:
        #     #Metrics for Binary Confusion Matrices
        #     precision = cf[1,1] / sum(cf[:,1])
        #     recall    = cf[1,1] / sum(cf[1,:])
        #     f1_score  = 2*precision*recall / (precision + recall)
        #     stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
        #         accuracy,precision,recall,f1_score)
    else:
        stats_text = ""

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False

    # MAKE THE HEATMAP VISUALIZATION
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories, ax=ax)

    if xyplotlabels:
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label' + stats_text)
    else:
        ax.set_xlabel(stats_text)
    
    if rotatelabels:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    return ax