# -*- coding: utf-8 -*-
# @Author : 小红牛
# 微信公众号：WdPython
import tkinter as tk
from tkinter import ttk
import pandas as pd

# 1.创建一个简单的DataFrame
data = {'诗人': ['李白', '苏轼', '李清照', '杜甫', '岳飞'],
        '性别': ['男', '男', '女', '男', '男'],
        '芳龄': [18, 26, 13, 15, 28],
        '朝代': ['唐', '北宋', '宋', '唐', '南宋'],
        '薪资': [9000, 7000, 8000, 5000, 7000]}

df = pd.DataFrame(data)
print('1.DataFrame原始数据'.center(50, '-'))
print(df)
# 2.创建主窗口
root = tk.Tk()
root.title("Treeview with DataFrame")
root.geometry('450x300')  # 窗口的宽度和高度

# 3.1.列表框数字+文本排序
def treeview_sort(tv, col, reverse):
    l = [(tv.set(k, col), k) for k in tv.get_children('')]
    # print(l)
    # print(l[0][0])
    # 1.处理数据里面的单位
    if '元' in l[0][0]:
        # 如果第一行的数据里存在 '元' 的文本
        l.sort(key=lambda t: int(t[0].replace('元', '')), reverse=reverse)  # 把单位去除后转数字再排序
    else:
        try:
            # 优先尝试数字排序
            l.sort(key=lambda t: int(t[0]), reverse=reverse)
        except:
            # 出错则普遍排序
            l.sort(reverse=reverse)
            # 这种排序根据首位字符来排序，不适合数字，会出现：1,11,2 这种不符合从大到小或从小到大的排序
    # print(l)
    # 移动数据
    for index, (val, k) in enumerate(l):
        # print(k)
        tv.move(k, '', index)

    tv.heading(col, command=lambda: treeview_sort(tv, col, not reverse))

# 3.2.df数据放到列表框
def dataframe_to_treeview(dfs, x1, y1, w, h, column_name='序号'):
    # 1.获取数据的列标题
    a = dfs.columns.values.tolist()
    a.insert(0, column_name)
    # 添加一个宽度列表，组成字典
    b = [80 for nums in range(len(a) - 1)]
    # [50, 80, 80, 80, 80, 80]
    b.insert(0, 50)
    df_titles = dict(zip(a, b))
    # print(df_titles)
    # 2.设置纵向滚动条
    xbar = tk.Scrollbar(root, orient='horizontal')
    xbar.place(x=x1, y=y1+h-3, width=w)
    ybar = tk.Scrollbar(root, orient='vertical')
    ybar.place(x=x1+w-3, y=y1, height=h)
    # 3.创建Treeview
    tree = ttk.Treeview(root, show='headings',
                     xscrollcommand = xbar.set,
                     yscrollcommand = ybar.set)

    tree['columns'] = list(df_titles)
    # 批量设置列属性
    for title in df_titles:
        # 加载列标题
        tree.heading(title, text=title)
        tree.column(title, width=df_titles[title], anchor='center')
        # 3.设置点击执行排序操作
        tree.heading(title, command=lambda _col=title: treeview_sort(tree, _col, False))

    # 遍历DataFrame的每一行，并将它们添加到Treeview中
    for index, row in dfs.iterrows():
        datas = row.tolist()
        datas.insert(0, index)
        # print(datas)
        # 添加行数据
        tree.insert('', 'end', text='', values=datas)
    # 将Treeview添加到主窗口
    tree.place(x=x1, y=y1, width=w, height=h)
    xbar.config(command=tree.xview)
    ybar.config(command=tree.yview)

# 创建两个表格
dataframe_to_treeview(df, 0, 20, 400, 100)
dataframe_to_treeview(df, 0, 170, 400, 100)
# 运行主循环
root.mainloop()
