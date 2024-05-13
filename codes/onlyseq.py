# -*- coding: utf-8 -*-
import sys
import get_dataset as data
import multi_model
from keras.utils import to_categorical
import time
import test_scores as score
from sklearn.metrics import roc_curve,auc
#from scipy import interp
import numpy as np
import matplotlib.pyplot as plt

tprs=[]
aucs=[]
mean_fpr=np.linspace(0,1,100)



import undersampling
def DeepW(file_name,kmer,seq_kernel,struc_kernel):
    model_path = './model/'
    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    # get training data and validation data
    print('get training data')
    train_file = './datasets/clip/' + file_name + '/30000/training_sample_0'
    train_data = data.dataset(train_file,kmer,seq_kernel,struc_kernel)
    train_label = train_data[1]

    # 随机下采样0
    seq_resampled,struc_resampled,label_resampled=undersampling.custom_random_undersampling(file_name,train_data[0][0],train_data[0][1],train_label,target_ratio=1.2,random_seed=42)

    ## ENN采样/聚类采样/CNN采样
    #seq_resampled,label_resampled,indices=undersampling.tomolinks_sampling(train_data[0][0],train_label)
    #struc_resampled=train_data[0][1][indices]


    #np.save(file_name + '_seq_feature.npy', seq_resampled)
    #np.save(file_name + '_struc_feature.npy', struc_resampled)
    #np.save(file_name + '_label.npy', label_resampled)





    train_indice, train_y, validation_indice, validation_y = data.split_training_validation(label_resampled)

    train_seq = seq_resampled[train_indice]
    val_seq = seq_resampled[validation_indice]
    train_struc = struc_resampled[train_indice]
    val_struc = struc_resampled[validation_indice]

    train_y = to_categorical(train_y, 2)
    val_y = to_categorical(validation_y, 2)


    # get test data
    print('get test data')
    test_file = './datasets/clip/' + file_name + '/30000/test_sample_0'
    test_data = data.dataset(test_file,kmer,seq_kernel,struc_kernel)
    test_seq = test_data[0][0]
    print(test_seq.shape)
    test_struc = test_data[0][1]
    print(test_struc.shape)
    test_label = test_data[1]
    test_y = to_categorical(test_label)

    #np.save(file_name[:10] + '_test_seq.npy', test_seq)
    #np.save(file_name[:10] + '_test_struc.npy', test_struc)
    #np.save(file_name[:10] + '_test_label.npy', test_y)

    # 3mer model
    if kmer == 1:
        print('1mer model training...')
        dw_model = multi_model.merged_BLSTM_CNN_1mer(train_seq.shape[1],train_struc.shape[1],seq_kernel, struc_kernel)
    elif kmer == 2:
        print('2mer model training...')
        #dw_model = multi_model.merged_BLSTM_CNN_2mer(train_seq.shape[1],train_struc.shape[1],seq_kernel, struc_kernel)
        dw_model = multi_model.onlyseq_model(train_seq.shape[1], seq_kernel)
    elif kmer == 3:
        print('3mer model training...')
        dw_model =multi_model.merged_BLSTM_CNN_3mer(train_seq.shape[1],train_struc.shape[1],seq_kernel, struc_kernel)
    dw_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    start_time=time.time()
    dw_model.fit(x=train_seq, y=train_y, batch_size=50, epochs=30, validation_data=(val_seq, val_y), verbose=1,shuffle=True)
    end_time=time.time()
    train_time=end_time-start_time
    print("训练时间：", train_time, 's')


    prob_val=dw_model.predict(val_seq)
    y_prob_val=prob_val[:,-1]
    # 计算验证集的ROC曲线
    fpr, tpr, thresholds = roc_curve(validation_y, y_prob_val)
    roc_auc = auc(fpr, tpr)
    #np.save(file_name[:10]+'_roc.npy',roc_auc)
    #np.save(file_name[:10]+'_fpr.npy',fpr)
    #np.save(file_name[:10] + '_tpr.npy', tpr)
    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of Dataset '+file_name)
    plt.legend(loc='lower right')
    #plt.savefig(file_name+'_ROC.png')
    plt.show()

    # save 2mer model
    dw_model.save(model_path+file_name[:10]+'_onlyseq.h5')
    prob = dw_model.predict(test_seq)
    y_pred_cnn = score.transfer_label_from_prob(prob[:,-1])
    y_prob = prob[:,-1]
    #precision,sensitivity,specificity,acc, balanced_acc, AUC,MCC
    p,sn,sp,acc, balanced_acc, AUC,mcc = score.calculate_performace(len(y_pred_cnn), y_pred_cnn, y_prob, test_y[:,-1])
    print(str(kmer)+'mer_model:\t'+'P:'+str(p)+'\t'+'SN:'+str(sn)+'\t'+'SP:'+str(sp)+'\t'+'ACC:'+str(acc)+'\t'+'balanced_acc:'+str(balanced_acc)+'\t'+'AUC:'+str(AUC)+'\t'+'MCC:'+str(mcc)+'\n')
    with open('./results/'+str(kmer)+'mer_model_kernel_'+str(seq_kernel)+'_'+str(struc_kernel)+'_BLSTM_compared.txt','a') as fw:
        fw.write(file_name+'_'+str(seq_kernel)+'_'+str(struc_kernel)+'\tP:'+str(p)+'\t'+'SN:'+str(sn)+'\t'+'SP:'+str(sp)+'\t'+'ACC:'+str(acc)+'\t'+'balanced_acc:'+str(balanced_acc)+'\t'+'AUC:'+str(AUC)+'\t'+'MCC:'+str(mcc)+'\n')


if __name__ == '__main__':
    kmer = 2
    seq_kernel = 4
    struc_kernel = 4
    dataset = '31_ICLIP_U2AF65_Hela_iCLIP_ctrl+kd_all_clusters'
    # 5_CLIPSEQ_AGO2_hg19 MCC=0?

    # 18-20缺少
    DeepW(dataset,kmer,seq_kernel,struc_kernel)



