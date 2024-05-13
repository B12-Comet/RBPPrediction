from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Treeview

import numpy as np
import test_scores as score
import re
from keras.models import load_model
from Bio import SeqIO
import pandas as pd
import tkinter.messagebox


root= Tk()

global df

# 定义格式化函数
def format_value(value, decimals):
    if decimals is not None:
        return "{:.{}f}".format(value, decimals)
    else:
        return int(value)




lb=Label(root,text='请选择测试文件：')
lb.grid(row=1,column=0,columnspan=1)
e=Entry(root,width=80)
e.grid(row=1,column=1,columnspan=1)


def open_file():


    file_path=filedialog.askopenfilename(title='请选择一个文件',
                                           filetypes=[('fasta', '.fa .fasta'), ('All Files', '*')])

    e.insert('insert',file_path)
    with open(file_path, 'r') as file:
        records = list(SeqIO.parse(file, 'fasta'))
    seqs = []
    head = []
    for record in records:
        seqs.append(record.seq)
        head.append(record.description)
    label = []
    for line in head:
        match = re.search(r'class:(\d+)', line)
        if match:
            class_number = match.group(1)
            class_number = int(class_number)
            if class_number==0:
                label.append('negative')
            else:
                label.append('positive')

    SeqNumber = len(records)

    sum = 0
    model = load_model('30_ICLIP_U.h5')

    test_seq = np.load('30_ICLIP_U_test_seq.npy')
    test_struc = np.load('30_ICLIP_U_test_struc.npy')
    test_y = np.load('30_ICLIP_U_test_label.npy')

    prob = model.predict([test_seq, test_struc])
    y_pred_cnn = score.transfer_label_from_prob(prob[:, -1])
    y_prob = prob[:, -1]  # 预测得到的概率
    y_prob=np.around(y_prob,decimals=4)
    y_pred_cnn = np.array(y_pred_cnn)  # 预测得到的0/1标签
    pred_label=[]
    for i in y_pred_cnn:
        if i==0:
            pred_label.append('negative')
        else:
            pred_label.append('positive')
    num = np.arange(start=1, stop=SeqNumber + 1, step=1)

    data = [num, label, y_prob, pred_label]
    global df
    df = pd.DataFrame(data)
    df = df.transpose()
    df.columns = ['序列序号', '实际标签', '概率', '预测标签']
    #df=df.applymap(lambda x:format_value(x,decimals=4) if x.name=='概率' else format_value(x,decimals=None))
    xscroll = Scrollbar(orient=HORIZONTAL)
    yscroll = Scrollbar(orient=VERTICAL)

    tree = Treeview(root,height=30,xscrollcommand=xscroll.set,yscrollcommand=yscroll.set,show='headings')
    tree['columns']=['序列序号','真实标签','概率','预测标签']
    tree.column('序列序号',width=100)
    tree.column('真实标签', width=100)
    tree.column('概率', width=100)
    tree.column('预测标签', width=100)
    tree.heading('序列序号',text='序列序号')
    tree.heading('真实标签', text='真实标签')
    tree.heading('概率', text='概率')
    tree.heading('预测标签', text='预测标签')
    xscroll.config(command=tree.xview)
    yscroll.config(command=tree.yview)

    for i in range(len(df)):
        tree.insert('',i,values=df.iloc[i,:].tolist())

    tree.grid(row=3, column=1)





btn = Button(root, text='打开文件', command=open_file)
btn.grid(row=1, column=2, columnspan=1)




result_lb=Label(root,text='预测结果：')
result_lb.grid(row=2)
def save():
    df.to_csv('predictor.csv', index=False,encoding='utf-8-sig')
    tkinter.messagebox.showinfo('提示','保存成功')
save_btn=Button(root,text='保存结果',command=save)
save_btn.grid(row=4,column=2)


root.title('Predictor')
root.geometry('800x700') # 这里的乘号不是 * ，而是小写英文字母 x

root.mainloop()





