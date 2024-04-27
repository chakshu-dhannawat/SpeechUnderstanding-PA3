import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def calculate_auc(labels, predictions):
    """
    Calculate the Area Under the ROC Curve (AUC).
    
    Args:
        labels (numpy.array): True labels.
        predictions (numpy.array): Predicted probabilities.

    Returns:
        float: AUC score.
    """
    auc = roc_auc_score(labels, predictions)
    return auc

def calculate_eer(labels, predictions):
    """
    Calculate the Equal Error Rate (EER).
    
    Args:
        labels (numpy.array): True labels.
        predictions (numpy.array): Predicted probabilities.

    Returns:
        float: EER score.
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    
    # Find the threshold for which false positive rate (FPR) equals false negative rate (FNR)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer
