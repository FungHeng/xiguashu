#-*- coding: utf-8 -*-
import pandas as pd
import pydotplus
from sklearn.preprocessing import LabelEncoder
#参数初始化
if __name__ == "__main__":
    inputfile = 'xiguashujuji3.txt'
    data = pd.read_table(inputfile, sep=',',index_col=0, encoding='ANSI')
    #数据是类别标签，要将它转化为数据
    le = LabelEncoder()  # 创建LabelEncoder()对象，用于序列化
    for col in data.columns:  # 为每一列序列化
        data[col] = le.fit_transform(data[col])

    x = data.iloc[:,:6].values.astype(int)
    y = data.iloc[:,8].values.astype(int)

    from sklearn.tree import DecisionTreeClassifier as DTC
    dtc = DTC(criterion='entropy') #建立决策树模型，基于信息熵
    dtc.fit(x, y) #训练模型

    #导入相关函数，可视化决策树
    #导入的结果是一个dot文件，需要安装Graphviz才能将它转化为pdf或png格式
    from sklearn.tree import export_graphviz
    x = pd.DataFrame(x)
    from sklearn.externals.six import StringIO
    x = pd.DataFrame(x)
    # with open("tree1.dot", 'w') as f:
    #   f = export_graphviz(dtc, feature_names = x.columns, out_file = f)

    dot_data = export_graphviz(dtc, out_file=None, feature_names=['seze', 'gendi', 'qiaosheng','wenli','qibu','chugan' ],class_names=['bad','good' ], filled=True, impurity=True, rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf('xigua.pdf')  # or png