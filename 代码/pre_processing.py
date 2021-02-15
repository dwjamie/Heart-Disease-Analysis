import pandas as pd

# 读入数据，并删除有空值的样本
data = pd.read_csv('heart_disease.csv').dropna(axis=0, how='any')

# 修改数据，方便理解
data.sex[data.sex == 0] = '女'
data.sex[data.sex == 1] = '男'

data.cp[data.cp == 1] = '典型心绞痛'
data.cp[data.cp == 2] = '非典型心绞痛'
data.cp[data.cp == 3] = '无心绞痛'
data.cp[data.cp == 4] = '渐进式疼痛'

data.fbs[data.fbs == 0] = '小于等于120mg/dl'
data.fbs[data.fbs == 1] = '大于120mg/dl'

data.restecg[data.restecg == 0] = '正常'
data.restecg[data.restecg == 1] = '患有ST-T波异常'
data.restecg[data.restecg == 2] = '左心室肥大'

data.exang[data.exang == 0] = '否'
data.exang[data.exang == 1] = '是'

data.slope[data.slope == 1] = '抬高'
data.slope[data.slope == 2] = '平缓'
data.slope[data.slope == 3] = '下降'

data.thal[data.thal == 3] = '正常'
data.thal[data.thal == 6] = '固定缺陷'
data.thal[data.thal == 7] = '可逆缺陷'

data.target[data.target != 0] = '患病'
data.target[data.target == 0] = '健康'

# 更改列名
data.columns = ['年龄', '性别', '胸痛类型', '静息血压', '血清总胆固醇', '空腹血糖', '静息心电图结果', '最大心率', '运动诱发的心绞痛',
                'ST段下降', '运动高峰ST段', '主要血管数', '地中海贫血', '诊断结果']

# 导出为csv数据
data.to_csv('heart_disease_preprocessed.csv', index=False)
