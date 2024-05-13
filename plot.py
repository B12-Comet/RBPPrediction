import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import time
fpr1=np.load('30_ICLIP_U_fpr.npy')
tpr1=np.load('30_ICLIP_U_tpr.npy')
roc_auc1=np.load('30_ICLIP_U_roc.npy')
fpr2=np.load('31_ICLIP_U_fpr.npy')
tpr2=np.load('31_ICLIP_U_tpr.npy')
roc_auc2=np.load('31_ICLIP_U_roc.npy')

plt.figure()
plt.plot(fpr1, tpr1, lw=2, label='Dataset 30 (AUC = {:.2f})'.format(roc_auc1))
plt.plot(fpr2, tpr2, lw=2, label='Dataset 31 (AUC = {:.2f})'.format(roc_auc2))


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Dataset 30 to 31')
plt.legend(loc='lower right')
plt.savefig('6_ROC.png')
plt.show()


