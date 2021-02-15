import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# 使中文能够在画出的图中正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def cross_val_report(name, clf, k=10, style='精简'):
    y_pred = cross_val_predict(clf, X, y, cv=k)
    if style == '精简':
        print(f'{name}: {round(100 * accuracy_score(y, y_pred), 2)}%')
    elif style == '完整':
        print('*' * 60)
        print(name)
        print(classification_report(y, y_pred, target_names=['健康', '患病'], digits=4))
        plot_confusion_matrix(y, y_pred, name)
    else:
        exit('参数输入错误！')


def plot_confusion_matrix(y, y_pred, name):
    plt.title(f'{name}混淆矩阵', y=-0.1)
    sns.heatmap(confusion_matrix(y, y_pred), cmap='PuBu', annot=True, fmt='g', cbar=False)
    plt.xlabel('预测分类')
    plt.xticks((0.5, 1.5), ('健康', '患病'))
    plt.ylabel('真实分类')
    plt.yticks((0.5, 1.5), ('健康', '患病'))
    ax = plt.gca()
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    plt.tight_layout()
    plt.savefig(f'./图片/混淆矩阵（交叉验证）/{name}混淆矩阵', dpi=300)
    plt.show()


if __name__ == '__main__':
    # 读入数据，设置哑变量
    data = pd.read_csv('heart_disease_preprocessed.csv')
    data = pd.get_dummies(data, drop_first=True)

    # 分开输入和输出，并对输入进行标准化
    X = scale(data.iloc[:, :-1])
    y = data.iloc[:, -1]

    # 所有待比较的模型
    names = ['Logistic回归']
    classifiers = [LogisticRegression(random_state=0)]
    for max_depth in (1, 2, 3, 5, 10, None):
        names.append(f'决策树（最大深度={max_depth}）')
        classifiers.append(DecisionTreeClassifier(max_depth=max_depth, random_state=0))
    for max_depth in (1, 2, 3, 5, 10, None):
        names.append(f'随机森林（最大深度={max_depth}）')
        classifiers.append(RandomForestClassifier(max_depth=max_depth, random_state=0))
    for kernel in ('linear', 'poly', 'rbf', 'sigmoid'):
        names.append(f'支持向量机（核函数={kernel}）')
        classifiers.append(SVC(kernel=kernel, random_state=0))
    for hidden_layer_sizes in ((50, ), (25, 25), (20, 20, 20), (10, 10, 10, 10, 10)):
        names.append(f'神经网络（隐层数量={len(hidden_layer_sizes)}，每个隐层的节点数={hidden_layer_sizes[0]}）')
        classifiers.append((MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=2000, random_state=0)))

    # 使用10折交叉验证的预测准确率比较各个模型的好坏
    for name, clf in zip(names, classifiers):
        cross_val_report(name, clf)

    # 进行10折交叉验证，输出各模型的完整分类报告，包括查准率、查全率、F1、混淆矩阵等
    for name, clf in zip(names, classifiers):
        cross_val_report(name, clf, style='完整')
