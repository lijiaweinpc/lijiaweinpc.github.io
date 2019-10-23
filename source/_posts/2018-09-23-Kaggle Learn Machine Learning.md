---
title: Kaggle Learn Machine Learning
date: 2018-09-23 18:49:10
tags: Kaggle
---

&emsp;&emsp;Kaggle网站教程，machine learning篇的笔记。

<!--more-->

&emsp;&emsp;直接就都记在了一个py文件里~

{% codeblock lang:python %}
# -*- coding: utf-8 -*-

"""Kaggle Learn->Machine Learning"""

"""Level 1"""

########################
## 1. How Models Work ##
########################

# 讲了一个决策树来判断房价的简单例子

#################################
## 2. Starting Your ML Project ##
#################################

# 用pandas来获取数据的描述性统计
import pandas as pd
# 把示例的data给下载了下来
melbourne_file_path = 'D:\Research\Datasets\kaggle\Learn-Machine Learning\melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
# describe会包含各series的非空计数、均值、标准差，最大最小以及四分位数。 
print(melbourne_data.describe())

#####################################
## 3. Selecting and Filtering Data ##
#####################################

# 打印列名
print(melbourne_data.columns)
# 选取其中的一列
melbourne_price_data = melbourne_data.Price
print(melbourne_price_data.head())
# 选取其中的多列
columns_of_interest = ['Landsize', 'BuildingArea']
two_columns_of_data = melbourne_data[columns_of_interest]
two_columns_of_data.describe()
# 从columns_of_interest的两列describe来看，下载的的数据集和原运行的数据略有出入，所以后续的模型结果也略有出入

#######################################
## 4. Choosing the Prediction Target ##
#######################################

# 指定预测的目标
y = melbourne_data.Price
# 选取自变量
melbourne_predictors = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_predictors]
# 使用sklearn决策树来建立模型
from sklearn.tree import DecisionTreeRegressor
# 模型定义
melbourne_model = DecisionTreeRegressor()
# 模型训练
melbourne_model.fit(X, y)
# 模型使用，这里谨使用训练数据来看一下用法
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))

#########################
## 5. Model Validation ##
#########################

# 使用平均绝对误差来评价模型
from sklearn.metrics import mean_absolute_error
predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
# 使用一组样本建立模型再计算MAE会过拟合，解决方案是拆分验证集
from sklearn.model_selection import train_test_split
# 拆分训练集和测试集，指定random_state为某个值保证每次的划分结果一致
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X, train_y)
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
      
#########################################################
## 6. Underfitting, Overfitting and Model Optimization ##
#########################################################

# 对我们建立的模型进行参数优化，在实践中决策树的深度选择每多一层类别*2，类别增加每个类别的样本就减少，层数过多会过拟合，少会欠拟合。
# 训练集的MAE随着深度增加会递减，测试集也就是我们的新数据到来时MAE随着深度增加会先递减（欠拟合）再递增（过拟合），我们想找的模型参数就在测试集驻点处。

# 为不同数量的树节点建立模型计算MAE
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

# 计算不同树节点的MAE，找驻点
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

#######################
## 7. Random Forests ##
#######################

# 随机森林将很多决策树的结果求平均俩获取超过单一决策树的准确率
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))

#################################
## 8. Submitting From A Kernel ##
#################################

# 讲解如何在kaggle提交模型预测结果

"""Level 2"""

################################
## 1. Handling Missing Values ##
################################

# 就取 melbourne_data.BuildingArea吧，查看有多少的空值
data = melbourne_data.BuildingArea.copy()
print(data.isnull().sum())

# 处理缺失最简单的是drop掉有空值的列
original_data = melbourne_data.copy()
data_without_missing_values = original_data.dropna(axis=1)
# 很多时候我们需要在测试集里同时把对应的列也drop掉，下面可以先找到有缺失的列再drop
cols_with_missing = [col for col in original_data.columns if original_data[col].isnull().any()]
redued_original_data = original_data.drop(cols_with_missing, axis=1)
# 更好的一个做法是插值，Imputer无法处理str，也需结合实际含义选择插值策略
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
data_with_imputed_values = my_imputer.fit_transform(original_data[['Car','BuildingArea','YearBuilt']])
# 插值大部分时间效果很好，但一个好的习惯还是新建一列记录哪些是我们插值的
new_data = original_data[['Car','BuildingArea','YearBuilt']].copy()
cols_with_missing = [col for col in new_data.columns if new_data[col].isnull().any()]
for col in cols_with_missing:
      new_data[col + '_was_missing'] = new_data[col].isnull()
my_imputer = Imputer()
new_data = my_imputer.fit_transform(new_data)

# 房价预测的完整示例：
import pandas as pd
melb_data = pd.read_csv(r'D:\Research\Datasets\kaggle\Learn-Machine Learning\melb_data.csv')
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
# 预测价格剥离出来
melb_target = melb_data.Price
melb_predictors = melb_data.drop(['Price'], axis=1)
# 为了简化说明只使用数值型的属性 
melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])
# 拆分训练集和测试集
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(melb_numeric_predictors, melb_target, train_size=0.7, test_size=0.3, random_state=0)
# 训练模型并计算MAE
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)
# 直接drop掉有缺失值的列
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)
print("直接drop掉有缺失值的列后建模的MAE:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))
# 对缺失值进行插值
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("对缺失值进行插值后建模的MAE:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))
# 将是否进行了插值作为新的特征一并训练模型
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()
cols_with_missing = (col for col in X_train.columns if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)
print("将是否进行了插值作为新的特征一并训练模型的MAE:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))

#####################################################
## 2. Using Categorical Data with One Hot Encoding ##
#####################################################

# 分类数据的标准处理方式是One-Hot Encoding,例子中换用了源
import pandas as pd
train_data = pd.read_csv(r'D:\Research\Datasets\kaggle\Learn-Machine Learning\train.csv')
test_data = pd.read_csv(r'D:\Research\Datasets\kaggle\Learn-Machine Learning\test.csv')

# 删掉缺失了价格（我们的目标）的行，并对缺失做简单插值
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
target = train_data.SalePrice
cols_with_missing = [col for col in train_data.columns 
                                 if train_data[col].isnull().any()]                                  
candidate_train_predictors = train_data.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)
candidate_test_predictors = test_data.drop(['Id'] + cols_with_missing, axis=1)

# 用不一样的值的数量以及col的dtype来近似判断分类数据字段
low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].nunique() < 10 and
                                candidate_train_predictors[cname].dtype == "object"]
numeric_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]

# 仅研究分类数据字段和数值型数据字段
my_cols = low_cardinality_cols + numeric_cols
train_predictors = candidate_train_predictors[my_cols]
test_predictors = candidate_test_predictors[my_cols]

# One-Hot Encoding,直接将object型的列全部编码
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)

# 下来比对有没有分类数据来做预测的MAE
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def get_mae(X, y):
    # sklearn习惯返回负值所以*-1
    return -1 * cross_val_score(RandomForestRegressor(50), X, y, scoring = 'neg_mean_absolute_error').mean()
predictors_without_categoricals = train_predictors.select_dtypes(exclude=['object'])
mae_without_categoricals = get_mae(predictors_without_categoricals, target)
mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, target)
print('drop分类数据的MAE: ' + str(int(mae_without_categoricals)))
print('One-Hot Encoding分类数据的MAE: ' + str(int(mae_one_hot_encoded)))

# 最后，记得预测的话对齐数据的处理方式
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
# align来对齐两个df的col顺序，left指定对不齐时以左边为准
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)
################
## 3. XGBoost ##
################

# XGBoost是GBDT算法的变种，通过集成模型来预测缩小误差
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
data = pd.read_csv(r'D:\Research\Datasets\kaggle\Learn-Machine Learning\train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)
my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

# 调用xgboost
from xgboost import XGBRegressor
my_model = XGBRegressor()
my_model.fit(train_X, train_y, verbose=False)
# 预测，计算MAE
predictions = my_model.predict(test_X)
from sklearn.metrics import mean_absolute_error
print("MAE: " + str(mean_absolute_error(predictions, test_y)))

# 模型优化调参
# n_estimators：模型迭代多少轮，迭代的越多越容易过拟合；
# early_stopping_rounds：多少轮模型没有优化后停止迭代，防止过拟合；经验设为5；
# learning_rate：参数移动到最优值的速度；学习率越小，迭代越多模型越好，但时间越慢；
# n_jobs：使用CPU核数。
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=True)
 
#################################
## 4. Partial Dependence Plots ##
#################################

# 局部依赖图展示的是特征和预测目标之间的关系；
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.preprocessing import Imputer

def get_some_data():
    cols_to_use = ['Distance', 'Landsize', 'BuildingArea']
    data = pd.read_csv(r'D:\Research\Datasets\kaggle\Learn-Machine Learning\melb_data.csv')
    y = data.Price
    X = data[cols_to_use]
    my_imputer = Imputer()
    imputed_X = my_imputer.fit_transform(X)
    return imputed_X, y

X, y = get_some_data()
# scikit-learn目前只支持GDBT绘制PDP，以后会支持全模型。
my_model = GradientBoostingRegressor()
my_model.fit(X, y)
my_plots = plot_partial_dependence(my_model,       
                                   features=[0, 1, 2], #想看到的特征列号
                                   X=X,           
                                   feature_names=['Distance', 'Landsize', 'BuildingArea'], 
                                   grid_resolution=10) #x轴的取值10

##################
## 5. Pipelines ##
##################

# 代码的规范性和清洁等问题
import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv(r'D:\Research\Datasets\kaggle\Learn-Machine Learning\melb_data.csv')
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
y = data.Price
train_X, test_X, train_y, test_y = train_test_split(X, y)

#使用make_pipeline让代码更清洁更专业
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())
my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(test_X)

#不使用pipeline的等价代码
my_imputer = Imputer()
my_model = RandomForestRegressor()
imputed_train_X = my_imputer.fit_transform(train_X)
imputed_test_X = my_imputer.transform(test_X)
my_model.fit(imputed_train_X, train_y)
predictions = my_model.predict(imputed_test_X)

#########################
## 6. Cross-Validation ##
#########################

# 交叉验证将数据集等分为多份，每次取一份为测试数据，其余为训练数据来训练模型。
# 与直接使用Train-Test Split相比，交叉验证使用所有数据训练模型，所以耗时会更长。
# 数据量小一些，计算时间可以接受的情况下推荐使用交叉验证，数据量大的话本身也不需要复用就Train-Test Split。
import pandas as pd
data = pd.read_csv(r'D:\Research\Datasets\kaggle\Learn-Machine Learning\melb_data.csv')
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
y = data.Price
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())

# negative MAE在别的地方用的不多，他可以对齐sklearn约定俗成的所用结果数值越大越好
#from sklearn.model_selection import cross_val_score
scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
print(scores)

# 模型整体的MAE,用来与其他模型作比对
print('整体的MAE %2f' %(-1 * scores.mean()))

#####################
## 7. Data Leakage ##
#####################

# 因果关系纰漏。
# Leaky Predictors：模型使用预测时拿不到的结果数据，模型显示效果极好但没有用。
# Leaky Validation Strategy：数据预处理有问题，比如你在未拆分训练测试集时就做了插值。
# 推荐使用scikit-learn Pipelines来降低出现Data Leakage的概率，对潜在有问题的特征还是要重点审视。
import pandas as pd
data = pd.read_csv(r'D:\Research\Datasets\kaggle\Learn-Machine Learning\AER_credit_card_data.csv', 
                   true_values = ['yes'],
                   false_values = ['no'])
print(data.head())

# 打印数据行列数
data.shape

# 建立模型，小样本使用交叉验证
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
y = data.card
X = data.drop(['card'], axis=1)
modeling_pipeline = make_pipeline(RandomForestClassifier())
cv_scores = cross_val_score(modeling_pipeline, X, y, scoring='accuracy')
print("Cross-val accuracy: %f" %cv_scores.mean())

# 去除潜在的问题特征，重建模型
potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X2 = X.drop(potential_leaks, axis=1)
cv_scores = cross_val_score(modeling_pipeline, X2, y, scoring='accuracy')
print("去除问题特征后的Cross-val accuracy: %f" %cv_scores.mean())
{% endcodeblock %}