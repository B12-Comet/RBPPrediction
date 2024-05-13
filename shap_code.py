import shap
import numpy as np
from keras.models import load_model
model_path='./30_ICLIP_U.h5'
model=load_model(model_path)
seq_fea=np.load('30_ICLIP_U_test_seq.npy')
struc_fea=np.load('30_ICLIP_U_test_struc.npy')
explainer = shap.DeepExplainer(model,[seq_fea,struc_fea])
shap.force_plot(explainer.expected_value[0])