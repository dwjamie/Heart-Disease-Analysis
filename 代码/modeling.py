import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 使中文能够在画出的图中正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def my_classification_report(name, clf):
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)[:, 1]

    print('*' * 60)
    print(name)
    print(classification_report(y, y_pred, target_names=['健康', '患病'], digits=4))
    plot_confusion_matrix(y, y_pred, name)

    # 画ROC曲线图
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    plt.title(f'{name}ROC曲线')
    plt.text(0.2, 0.6, f'AUC = {round(auc(fpr, tpr), 4)}', fontsize=15)
    plt.plot(fpr, tpr)
    plt.plot((0, 1), (0, 1), linestyle='--')
    plt.tight_layout()
    plt.savefig(f'./图片/ROC曲线图/{name}ROC曲线图', dpi=300)
    plt.show()


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
    plt.savefig(f'./图片/混淆矩阵/{name}混淆矩阵', dpi=300)
    plt.show()


if __name__ == '__main__':
    # 读入数据，设置哑变量
    data = pd.read_csv('heart_disease_preprocessed.csv')
    data = pd.get_dummies(data, drop_first=True)

    # 分开输入和输出，并对输入进行标准化
    X = scale(data.iloc[:, :-1])
    y = data.iloc[:, -1]

    names = ['Logistic回归', '随机森林（最大深度=2）', '随机森林（最大深度=3）']
    classifiers = [LogisticRegression(random_state=0),
                   RandomForestClassifier(max_depth=2, random_state=0),
                   RandomForestClassifier(max_depth=3, random_state=0)]

    for name, clf in zip(names, classifiers):
        my_classification_report(name, clf)

    feature_names = data.columns[:-1].values
    importances_list = [classifiers[0].coef_.ravel(),
                        classifiers[1].feature_importances_,
                        classifiers[2].feature_importances_]

    for name, importances in zip(names, importances_list):
        plt.title(f'{name}特征重要性分布')
        sns.barplot(x=importances, y=feature_names, palette=['#2077B4'])
        plt.tight_layout()
        plt.savefig(f'./图片/特征重要性分布/{name}特征重要性分布', dpi=300)
        plt.show()
