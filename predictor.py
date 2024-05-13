
from Bio import SeqIO
from tkinter import filedialog
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import test_scores as score
import re
from keras.models import load_model


model = load_model('30_ICLIP_U.h5')

test_seq=np.load('30_ICLIP_U_test_seq.npy')
test_struc=np.load('30_ICLIP_U_test_struc.npy')
test_y=np.load('30_ICLIP_U_test_label.npy')

prob = model.predict([test_seq,test_struc])
y_pred_cnn = score.transfer_label_from_prob(prob[:,-1]) #预测得到的0/1标签
y_prob = prob[:,-1]
y_pred_cnn=np.array(y_pred_cnn)
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(filetypes=[('Text files', '*.fasta')])

with open(file_path, 'r') as file:
    records = list(SeqIO.parse(file, 'fasta'))

seqs=[]
head=[]
for record in records:
    seqs.append(record.seq)
    head.append(record.description)
label=[]
for line in head:
    match = re.search(r'class:(\d+)', line)
    if match:
        class_number = match.group(1)
        label.append([class_number])
SeqNumber = len(records)
n = 0

Name = []
testD1 = []
position = []
n = 0



sum=0



for i in range(SeqNumber):
    print('Seq '+str(i+1)+'\t'+head[i])
    if(y_pred_cnn[i]==0):
        if(test_y[:,-1][i]==1):
            print('\t\tpossibility:'+str(prob[:,-1][i])+'\t'+'\tnegative')# 预测错误
        else:
            sum=sum+1
            print('\t\tpossibility:'+str(prob[:,-1][i])+'\t'+'\033[31m\tnegative\033[0m')

    else:
        if (test_y[:,-1][i] == 0):
            print('\t\tpossiblity:'+str(prob[:,-1][i])+'\t'+'\tpositive') #预测错误
        else:
            sum=sum+1
            print('\t\tpossibility:'+str(prob[:,-1][i])+'\t'+'\033[31m\tpositive\033[0m')

print('total correct numbers:'+str(sum)+'/'+str(SeqNumber))
print('end')