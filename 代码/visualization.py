import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic

# 使中文能够在画出的图中正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读入数据
data = pd.read_csv('heart_disease_preprocessed.csv')

# 正式画图
# ---------------图0 数据概览图---------------
sns.pairplot(data)
plt.tight_layout()
plt.savefig('./图片/统计图表/图0_数据概览', dpi=300)
plt.show()

# ---------------图1 患病与健康的年龄分布条形图---------------
sns.countplot(x='年龄', hue='诊断结果', data=data, palette=['green', 'red'], alpha=0.75)
plt.title('年龄分布')
plt.ylabel('频数')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('./图片/统计图表/图1_年龄分布', dpi=300)
plt.show()

# ---------------图2，图3 患病/健康的胸痛类型饼状图---------------
labels = '典型心绞痛', '非典型心绞痛', '渐进式疼痛', '无心绞痛'
sizes = [7, 9, 103, 18]
explode = (0.01, 0.01, 0.01, 0.01)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%')
plt.title('患病者的胸痛类型占比')
plt.tight_layout()
plt.savefig('./图片/统计图表/图2_患病者的胸痛类型占比', dpi=300)
plt.show()

labels = '典型心绞痛', '非典型心绞痛', '渐进式疼痛', '无心绞痛'
sizes = [16, 40, 39, 65]
explode = (0.01, 0.01, 0.01, 0.01)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%')
plt.title('健康者的胸痛类型占比')
plt.tight_layout()
plt.savefig('./图片/统计图表/图3_健康者的胸痛类型占比', dpi=300)
plt.show()

# ---------------图4 患病/健康的静息血压箱线图---------------
sns.catplot(x='诊断结果', y='静息血压', kind="box", data=data)
plt.title('静息血压分布')
plt.xlabel(None)
plt.tight_layout()
plt.savefig('./图片/统计图表/图4_静息血压分布', dpi=300)
plt.show()

# ---------------图5 患病/健康的最大心率分布箱线图---------------
sns.catplot(x='诊断结果', y='最大心率', kind="box", data=data)
plt.title('最大心率分布')
plt.xlabel(None)
plt.tight_layout()
plt.savefig('./图片/统计图表/图5_最大心率分布', dpi=300)
plt.show()

# ---------------图6 患病/健康的静息心电图的条形图---------------
sns.countplot(x='静息心电图结果', hue='诊断结果', data=data, palette=['royalblue', 'red'], alpha=0.75)
plt.title('静息心电图结果分布')  # 设置标题
plt.ylabel('频数')  # 设置纵轴标签
plt.legend(loc='upper right')  # 手动调整图例的位置
plt.tight_layout()  # 自动美化图片排版
plt.savefig('./图片/统计图表/图6_静息心电图结果分布', dpi=300)  # 保存图像到工作路径，并设置分辨率
plt.show()  # 显示图像

# ---------------图7 运动诱发心绞痛类型与心脏病的马赛克图---------------
data1 = {('不是运动诱发的心绞痛', '患病'): 63, ('是运动诱发的心绞痛', '患病'): 74, ('不是运动诱发的心绞痛', '健康'): 137, ('是运动诱发的心绞痛', '健康'): 23}
mosaic(data1, title='运动诱发心绞痛和心脏病关系')
plt.tight_layout()
plt.savefig('./图片/统计图表/图7_运动诱发心绞痛和心脏病关系', dpi=300)
plt.show()

# ---------------图8 诊断结果性别分布条形图---------------
sns.countplot(x='性别', hue='诊断结果', data=data, palette=['green', 'red'], alpha=0.75)
plt.title('诊断结果的性别分布条形图')  # 设置标题
plt.ylabel('频数')  # 设置纵轴标签
plt.legend(loc='upper right')  # 手动调整图例的位置
plt.tight_layout()  # 自动美化图片排版
plt.savefig('./图片/统计图表/图8_诊断结果的性别分布', dpi=300)  # 保存图像到工作路径，并设置分辨率
plt.show()  # 显示图像

# ---------------图9 诊断结果性别分布饼状图（男性）---------------
dataman = [112, 89]
label = ["患病", "健康"]
color = ["red", "green"]
plt.pie(x=dataman, labels=label, colors=color, autopct="%.0f%%")
plt.axis("equal")
plt.title("男性患病与健康比例图")
plt.tight_layout()
plt.savefig('./图片/统计图表/图9_男性患病与健康比例图', dpi=300)  # 保存图像到工作路径，并设置分辨率
plt.show()

# ---------------图10 诊断结果性别分布饼状图（女性）---------------
datawoman = [25, 71]
color = ["red", "green"]
plt.pie(x=datawoman, labels=label, colors=color, autopct="%.0f%%")
plt.axis("equal")
plt.title("女性患病与健康比例图")
plt.tight_layout()
plt.savefig('./图片/统计图表/图10_女性患病与健康比例图', dpi=300)
plt.show()

# ---------------图11 不同血糖浓度与诊断结果分布条形图---------------
sns.countplot(x='空腹血糖', hue='诊断结果', data=data, palette=['royalblue', 'red'], alpha=0.7)
plt.title('不同血糖浓度与诊断结果分布条形图')
plt.xlabel("空腹血糖浓度")  # 取消横轴标签
plt.ylabel('频数')
plt.tight_layout()
plt.savefig('./图片/统计图表/图11_不同血糖浓度与诊断结果分布条形图', dpi=300)
plt.show()

# ---------------图12 不同血糖浓度与诊断结果的饼状图(<=120mg/dl)---------------
data120lower = [114, 137]
label = ["患病", "健康"]
color = ["red", "green"]
plt.pie(x=data120lower, labels=label, colors=color, autopct="%.0f%%")
plt.axis("equal")
plt.title("空腹血糖浓度小于等于120mg/dl患病与健康比例图")
plt.tight_layout()
plt.savefig('./图片/统计图表/图12_空腹血糖浓度小于等于120mgdl患病与健康比例图', dpi=300)
plt.show()

# ---------------图13 不同血糖浓度与诊断结果的饼状图(>120mg/dl)---------------
data120higher = [20, 23]
label = ["患病", "健康"]
color = ["red", "green"]
plt.pie(x=data120higher, labels=label, colors=color, autopct="%.0f%%")
plt.axis("equal")
plt.title("空腹血糖浓度>120mg/dl患病与健康比例图")
plt.savefig('./图片/统计图表/图13_空腹血糖浓度大于120mgdl患病与健康比例图', dpi=300)
plt.tight_layout()
plt.show()

# ---------------图14 不同诊断结果的血清总胆固醇箱线图---------------
sns.boxplot(x='诊断结果', y='血清总胆固醇', data=data)
plt.title('不同诊断结果的血清总胆固醇箱线图')
plt.xlabel("诊断结果")
plt.tight_layout()
plt.savefig('./图片/统计图表/图14_不同诊断结果的血清总胆固醇箱线图', dpi=300)
plt.show()

# ---------------图15 胆固醇平均值对比图---------------
sns.catplot(x="性别", y="血清总胆固醇", hue="诊断结果", kind="bar", palette=['green', 'red'], data=data)
plt.title('胆固醇平均值对比')
plt.savefig('./图片/统计图表/图15_胆固醇平均值对比', dpi=300)
plt.show()

# ---------------图16 年龄与血清总胆固醇散点图---------------
sns.regplot(x="年龄", y="血清总胆固醇", data=data)
plt.title("年龄与血清总胆固醇散点图")
plt.xlabel("年龄")
plt.ylabel("血清总胆固醇")
plt.savefig('./图片/统计图表/图16_年龄与血清总胆固醇散点图', dpi=300)
plt.show()

# ---------------图17 年龄与血清总胆固醇及确诊散点图---------------
sns.relplot(x="年龄", y="血清总胆固醇", hue="诊断结果", data=data)
plt.title("年龄与血清总胆固醇及确诊散点图")
plt.xlabel("年龄")
plt.ylabel("血清总胆固醇")
plt.savefig('./图片/统计图表/图17_年龄与血清总胆固醇及确诊散点图', dpi=300)
plt.show()

# ---------------图18 运动高峰ST段分布条形图---------------
sns.countplot(x='运动高峰ST段', hue='诊断结果', data=data, palette=['green', 'red'], alpha=0.75)
plt.title('运动高峰ST段分布')
plt.ylabel('频数')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('./图片/统计图表/图18_运动高峰ST段分布', dpi=300)
plt.show()

# ---------------图19，图20 运动高峰ST段分布箱线图---------------
labels = ('下降', '平缓', '抬高')

x = [data[data.诊断结果 == '健康'].运动高峰ST段.value_counts()[label] for label in labels]
plt.title('健康者的运动高峰ST段分布图')
plt.pie(x, labels=labels, autopct='%.1f%%')
plt.tight_layout()
plt.savefig('./图片/统计图表/图19_健康者的运动高峰ST段分布图', dpi=300)
plt.show()

x = [data[data.诊断结果 == '患病'].运动高峰ST段.value_counts()[label] for label in labels]
plt.title('患病者的运动高峰ST段分布图')
plt.pie(x, labels=labels, autopct='%.1f%%')
plt.tight_layout()
plt.savefig('./图片/统计图表/图20_患病者的运动高峰ST段分布图', dpi=300)
plt.show()

# ---------------图21 ST段下降分布箱线图---------------
sns.boxplot(x='诊断结果', y='ST段下降', data=data)
plt.title('ST段下降分布')
plt.xlabel(None)
plt.tight_layout()
plt.savefig('./图片/统计图表/图21_ST段下降分布', dpi=300)
plt.show()

# ---------------图22 不同胸痛类型的ST段下降分布箱线图---------------
sns.boxplot(x='胸痛类型', y='ST段下降', hue='诊断结果', data=data)
plt.title('不同胸痛类型的ST段下降分布')
plt.tight_layout()
plt.savefig('./图片/统计图表/图22_不同胸痛类型的ST段下降分布', dpi=300)
plt.show()

# ---------------图26，图30 地中海贫血/主要血管数马赛克图---------------
data = pd.read_csv('heart_disease_preprocessed.csv')
mosaic(data, ['地中海贫血', '诊断结果'], horizontal=False)
plt.savefig('./图片/统计图表/图26_地中海贫血马赛克图.jpg', dpi=1000)
plt.clf()
mosaic(data, ['主要血管数', '诊断结果'], horizontal=False)
plt.savefig('./图片/统计图表/图30_主要血管数马赛克图.jpg', dpi=1000)
plt.clf()

# ---------------图24，图25——地中海贫血饼图---------------
data0 = data[data['诊断结果'] == '健康']
data1 = data[data['诊断结果'] == '患病']
X0 = data0.loc[:, ['主要血管数', '地中海贫血']]
X1 = data1.loc[:, ['主要血管数', '地中海贫血']]

data = X0['地中海贫血'].value_counts()
labels = ['正常', '可逆缺陷', '固定缺陷']
plt.axes(aspect='equal')
plt.pie(x=data, labels=labels, colors=['orangered', 'yellow', 'darkseagreen'],
        autopct='%.1f%%', pctdistance=0.6, labeldistance=1.1, textprops={'fontsize': 12, "color": "black"})
plt.title('健康人群地中海贫血', fontweight='heavy', horizontalalignment='center', fontsize=18)
plt.savefig('./图片/统计图表/图24_健康人群地中海贫血.jpg', dpi=1000)
plt.clf()

data = X1['地中海贫血'].value_counts()
labels = ['正常', '可逆缺陷', '固定缺陷']
plt.axes(aspect='equal')
plt.pie(x=data, labels=labels, colors=['orangered', 'yellow', 'darkseagreen'],
        autopct='%.1f%%', pctdistance=0.6, labeldistance=1.1, textprops={'fontsize': 12, "color": "black"})
plt.title('患病人群地中海贫血', fontweight='heavy', horizontalalignment='center', fontsize=18)
plt.savefig('./图片/统计图表/图25_患病人群地中海贫血.jpg', dpi=1000)
plt.clf()

# ---------------图28，图29——主要血管数饼图---------------
data = X1['主要血管数'].value_counts()
labels = ['0', '1', '2', '3']
plt.axes(aspect='equal')
plt.pie(x=data, labels=labels, colors=['orangered', 'yellow', 'darkseagreen', 'cornflowerblue'],
        autopct='%.1f%%', pctdistance=0.6, labeldistance=1.1, textprops={'fontsize': 12, "color": "black"})
plt.title('患病人群主要血管数', fontweight='heavy', horizontalalignment='center', fontsize=18)
plt.savefig('./图片/统计图表/图28_患病人群主要血管数.jpg', dpi=1000)
plt.clf()

data = X0['主要血管数'].value_counts()
labels = ['0', '1', '2', '3']
plt.axes(aspect='equal')
plt.pie(x=data, labels=labels, autopct='%.1f%%', colors=['orangered', 'yellow', 'darkseagreen', 'cornflowerblue'],
        pctdistance=0.6, labeldistance=1.1, textprops={'fontsize': 12, "color": "black"})
plt.title('健康人群主要血管数', fontweight='heavy', horizontalalignment='center', fontsize=18)
plt.savefig('./图片/统计图表/图29_健康人群主要血管数.jpg', dpi=1000)
plt.clf()

# ---------------图23，图27——地中海贫血/主要血管数条形图---------------
x = np.arange(3)
y = X0['地中海贫血'].value_counts()
y1 = X1['地中海贫血'].value_counts()
bar_width = 0.35

tick_label = ['正常', '可逆缺陷', '固定缺陷']
plt.bar(x, y, bar_width, align="center", color="c", label="健康", alpha=0.5)
plt.bar(x + bar_width, y1, bar_width, color="b", align="center", label="患病", alpha=0.5)
plt.xlabel("类型")
plt.ylabel("人数")
plt.xticks(x + bar_width / 2, tick_label)
plt.legend()
plt.title('地中海贫血', fontweight='heavy', horizontalalignment='center', fontsize=18)
plt.savefig('./图片/统计图表/图23_地中海贫血条形图.jpg', dpi=1000)
plt.clf()

x = np.arange(4)
y = X0['主要血管数'].value_counts()
y1 = X1['主要血管数'].value_counts()
bar_width = 0.35
tick_label = ['0', '1', '2', '3']
plt.bar(x, y, bar_width, align="center", color="c", label="健康", alpha=0.5)
plt.bar(x + bar_width, y1, bar_width, color="b", align="center", label="患病", alpha=0.5)
plt.xlabel("类型")
plt.ylabel("人数")
plt.xticks(x + bar_width / 2, tick_label)
plt.legend()
plt.title('主要血管数', fontweight='heavy', horizontalalignment='center', fontsize=18)
plt.savefig('./图片/统计图表/图27_主要血管数条形图.jpg', dpi=1000)
