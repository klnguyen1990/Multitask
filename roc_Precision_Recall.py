#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


def compute_ROC_auc(y_label,y_predicted,n_classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_predicted[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return roc_auc, fpr, tpr


# In[1]:


def plot_ROC_curve(fpr,tpr,roc_auc,classe,color="red"):
    lw = 2
    plt.plot(
        fpr[classe],
        tpr[classe],
        color=color,
        lw=lw,
        label="Classe "+str(classe)+" (area = %0.2f)" % roc_auc[classe],
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Courbe de ROC")
    plt.legend(loc="lower right")
    
    #plt.show()


# In[ ]:


def compute_precision_recall(y_label,y_predicted,n_classes):
    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_label[:, i], y_predicted[:, i])
        average_precision[i] = average_precision_score(y_label[:, i], y_predicted[:, i])
    return precision, recall,average_precision


# In[ ]:


def plot_precision_recall_curve(precision, recall, average_precision,n_classes,color):
    plt.clf()
    #plt.figure(figsize=(20,30))
    for i in range(n_classes):
        plt.plot(recall[i], precision[i],
                color=color[i],
                label='Class {0} (area = {1:0.2f})'.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    #plt.rc('font', size=18)
    plt.show()


# In[ ]:


def TruePositif(y_pred, y_label) :
    TP = np.sum(y_pred * y_label)
    return TP
def TrueNegatif(y_pred, y_label) :
    TN = 0
    for i in range(y_pred.shape[0] - 1) :
        if y_pred[i] == y_label[i] and y_pred[i] == 0 :
            TN = TN + 1
    return TN
def FauxPositif(y_pred, y_label) :
    FP = 0
    for i in range(y_pred.shape[0] - 1) :
        if y_pred[i] == 1 and y_label[i] == 0 :
            FP = FP + 1
    return FP
def FauxNegatif(y_pred, y_label) :
    FN = 0
    for i in range(y_pred.shape[0] - 1) :
        if y_pred[i] == 0 and y_label[i] == 1 :
            FN = FN + 1
    return FN


# In[ ]:


def precision_recall_score(classe_pred,val_label_classe):
    TP = TruePositif(classe_pred, val_label_classe)
    FP = FauxPositif(classe_pred, val_label_classe)
    TN = TrueNegatif(classe_pred, val_label_classe)
    FN = FauxNegatif(classe_pred, val_label_classe)
    print('Precision High Risque= ', round(TP/(TP+FP)*100, 2), '%')
    print('Rappel High Risque = ', round(TP/(TP+FN)*100, 2), '%')
    print('Precision Low Risque= ', round(TN/(TN+FN)*100, 2), '%')
    print('Rappel Low Risque = ', round(TN/(TN+FP)*100, 2), '%')


# In[ ]:


def true_false_positive(threshold_vector, y_test):
    true_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 1)
    true_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 0)
    false_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 0)
    false_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 1)

    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())

    return tpr, fpr


# In[ ]:


def roc_from_scratch(probabilities, y_test, partitions=100):
    roc = np.array([])
    for i in range(partitions + 1):
        
        threshold_vector = np.greater_equal(probabilities, i / partitions).astype(int)
        tpr, fpr = true_false_positive(threshold_vector, y_test)
        roc = np.append(roc, [fpr, tpr])
        
    return roc.reshape(-1, 2)

