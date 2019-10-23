---
title: lasso+randomforest预测房价
date: 2018-11-14 20:36:13
tags: Kaggle
---

&emsp;&emsp;Kaggle的练习比赛——House Prices Advanced Regression Techniques,使用了lasso和randomforest来预测房价，误差0.12772（37%）。留底源码。

<!--more-->

&emsp;&emsp;不知道为什么开了代理kaggle的kernel都提交不了，郁闷啊，在这边留底一份源码吧，模型结果有待改进但是整体框架应该是比较全面的。
{% codeblock lang:python %}
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:41:10 2018

主要参考：
https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

@author: jiawei
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew

##############
## settings ##
##############

# set seaborn color and style
sns.set_palette('hls')
sns.set_style('darkgrid')

# limiting floats output to 3 decimal points
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

# ignore annoying warning (from sklearn and seaborn)
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

###############
## load data ##
###############

# get datasets
train = pd.read_csv(r'D:\Research\Datasets\kaggle\House Prices Advanced Regression Techniques\train.csv')
test = pd.read_csv(r'D:\Research\Datasets\kaggle\House Prices Advanced Regression Techniques\test.csv')
sample_submission = pd.read_csv(r'D:\Research\Datasets\kaggle\House Prices Advanced Regression Techniques\sample_submission.csv')

# save and drop ids
train_id = train.Id
test_id = test.Id
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

##############################
## target variable analysis ##
##############################

# comparing target to norm
sns.distplot(train['SalePrice'], fit=norm)
(mu, sigma) = norm.fit(train['SalePrice'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

# transform and make target more normally distributed use log1p:log(1+x)
train["SalePrice"] = np.log1p(train["SalePrice"])

##########################
## features engineering ##
##########################

# correlation map
corrmat = train.corr()
plt.subplots(figsize=(10,10))
sns.heatmap(corrmat, vmax=0.8, square=True)

# Top10 correlation matrix
cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# scatterplot top correlate features
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars']
sns.pairplot(train[cols], size = 2.5)
plt.show()

# box plot categorical features
data = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y='SalePrice', data=data)

# delete outliers
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.xlabel('GrLivArea', fontsize=10)
plt.ylabel('SalePrice', fontsize=10)
plt.show()
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<13)].index)

# explore train and test data together
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)

# check every feature, impute missing values, encode categorical features
tocheck_features = (pd.DataFrame(all_data.dtypes).join(corrmat.SalePrice)).rename(columns = {0:'type', 'SalePrice': 'CorrToTarget'})
'''
MSSubClass: Identifies the type of dwelling involved in the sale.	
        20	1-STORY 1946 & NEWER ALL STYLES
        30	1-STORY 1945 & OLDER
        40	1-STORY W/FINISHED ATTIC ALL AGES……
        so it's categorical feature in int, no order.
'''
print('nums of na: ', all_data['MSSubClass'].isnull().sum())
all_data['MSSubClass'] = all_data['MSSubClass'].astype(object)  
tocheck_features.drop('MSSubClass', inplace = True)
'''
MSZoning: Identifies the general zoning classification of the sale.		
       A	Agriculture
       C	Commercial
       FV	Floating Village Residential……
       categorical feature, 4 values missing, fill with most common value.
'''
print('nums of na: ', all_data['MSZoning'].isnull().sum())
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
tocheck_features.drop('MSZoning', inplace = True)
'''
LotFrontage: Linear feet of street connected to property
      486 values missing, fill with Neighborhood'median value.
'''
print('nums of na: ', all_data['LotFrontage'].isnull().sum())
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
tocheck_features.drop('LotFrontage', inplace = True)
'''
LotArea: Lot size in square feet
'''
print('nums of na: ', all_data['LotArea'].isnull().sum())
tocheck_features.drop('LotArea', inplace = True)
'''
Street: Type of road access to property
       Grvl	Gravel	
       Pave	Paved
'''
print('nums of na: ', all_data['Street'].isnull().sum())
tocheck_features.drop('Street', inplace = True)
'''
Alley: Type of alley access to property
       Grvl	Gravel
       Pave	Paved
       NA 	No alley access
       2721 values missing, fill with NotExist.
'''
print('nums of na: ', all_data['Alley'].isnull().sum())
all_data['Alley'] = all_data['Alley'].fillna('NotExist')
tocheck_features.drop('Alley', inplace = True)
'''
LotShape: General shape of property
       Reg	Regular	
       IR1	Slightly irregular
       IR2	Moderately Irregular
       IR3	Irregular
'''
print('nums of na: ', all_data['LotShape'].isnull().sum())
tocheck_features.drop('LotShape', inplace = True)
'''
LandContour: Flatness of the property
       Lvl	Near Flat/Level	
       Bnk	Banked - Quick and significant rise from street grade to building……       
'''
print('nums of na: ', all_data['LandContour'].isnull().sum())
tocheck_features.drop('LandContour', inplace = True)
'''
Utilities: Type of utilities available		
       AllPub	All public Utilities (E,G,W,& S)	
       NoSewr	Electricity, Gas, and Water (Septic Tank)
       NoSeWa	Electricity and Gas Only……
'''
print('nums of na: ', all_data['Utilities'].isnull().sum())
all_data['Utilities'] = all_data['Utilities'].fillna(all_data['Utilities'].mode()[0])
tocheck_features.drop('Utilities', inplace = True)
'''
LotConfig: Lot configuration
       Inside	Inside lot
       Corner	Corner lot
       CulDSac	Cul-de-sac……
'''
print('nums of na: ', all_data['LotConfig'].isnull().sum())
tocheck_features.drop('LotConfig', inplace = True)
'''
LandSlope: Slope of property		
       Gtl	Gentle slope
       Mod	Moderate Slope	
       Sev	Severe Slope
'''
print('nums of na: ', all_data['LandSlope'].isnull().sum())
tocheck_features.drop('LandSlope', inplace = True)
'''
Neighborhood: Physical locations within Ames city limits
       Blmngtn	Bloomington Heights
       Blueste	Bluestem
       BrDale	Briardale
'''
print('nums of na: ', all_data['Neighborhood'].isnull().sum())
tocheck_features.drop('Neighborhood', inplace = True)
'''
Condition1: Proximity to various conditions	
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal……
'''
print('nums of na: ', all_data['Condition1'].isnull().sum())
tocheck_features.drop('Condition1', inplace = True)
'''
Condition2: Proximity to various conditions (if more than one is present)	
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal……	
'''
print('nums of na: ', all_data['Condition2'].isnull().sum())
tocheck_features.drop('Condition2', inplace = True)
'''
BldgType: Type of dwelling		
       1Fam	Single-family Detached	
       2FmCon	Two-family Conversion; originally built as one-family dwelling
       Duplx	Duplex……
'''       
print('nums of na: ', all_data['BldgType'].isnull().sum())
tocheck_features.drop('BldgType', inplace = True)       
'''       
HouseStyle: Style of dwelling	
       1Story	One story
       1.5Fin	One and one-half story: 2nd level finished
       1.5Unf	One and one-half story: 2nd level unfinished……
'''
print('nums of na: ', all_data['HouseStyle'].isnull().sum())
tocheck_features.drop('HouseStyle', inplace = True)
'''
OverallQual: Rates the overall material and finish of the house
       10	Very Excellent
       9	Excellent
       8	Very Good……
'''
print('nums of na: ', all_data['OverallQual'].isnull().sum())
tocheck_features.drop('OverallQual', inplace = True)
'''
OverallCond: Rates the overall condition of the house
       10	Very Excellent
       9	Excellent
       8	Very Good……       
'''
print('nums of na: ', all_data['OverallCond'].isnull().sum())
tocheck_features.drop('OverallCond', inplace = True)
'''
YearBuilt: Original construction date
'''
print('nums of na: ', all_data['YearBuilt'].isnull().sum())
tocheck_features.drop('YearBuilt', inplace = True)
'''
YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
'''
print('nums of na: ', all_data['YearRemodAdd'].isnull().sum())
tocheck_features.drop('YearRemodAdd', inplace = True)
'''
RoofStyle: Type of roof
       Flat	Flat
       Gable	Gable
       Gambrel	Gabrel (Barn)……
'''
print('nums of na: ', all_data['RoofStyle'].isnull().sum())
tocheck_features.drop('RoofStyle', inplace = True)
'''
RoofMatl: Roof material
       ClyTile	Clay or Tile
       CompShg	Standard (Composite) Shingle
       Membran	Membrane……
'''
print('nums of na: ', all_data['RoofMatl'].isnull().sum())
tocheck_features.drop('RoofMatl', inplace = True)
'''
Exterior1st: Exterior covering on house
       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common……
       1 value missing, fill with most common value.
'''
print('nums of na: ', all_data['Exterior1st'].isnull().sum())
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
tocheck_features.drop('Exterior1st', inplace = True)      
'''
Exterior2nd: Exterior covering on house (if more than one material)
       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common……
       1 value missing, fill with most common value.
'''
print('nums of na: ', all_data['Exterior2nd'].isnull().sum())
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
tocheck_features.drop('Exterior2nd', inplace = True)  
'''
MasVnrType: Masonry veneer type
       BrkCmn	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block……     
       24 values missing, fill with most common value.
'''
print('nums of na: ', all_data['MasVnrType'].isnull().sum())
all_data['MasVnrType'] = all_data['MasVnrType'].fillna(all_data['MasVnrType'].mode()[0])
tocheck_features.drop('MasVnrType', inplace = True) 
'''
MasVnrArea: Masonry veneer area in square feet
      23 values missing, fill with most common value.
'''
print('nums of na: ', all_data['MasVnrArea'].isnull().sum())
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
tocheck_features.drop('MasVnrArea', inplace = True) 
'''
ExterQual: Evaluates the quality of the material on the exterior 		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical……
       ordered categorical feature, change to scores. 
'''
print('nums of na: ', all_data['ExterQual'].isnull().sum())
all_data['ExterQual'] = all_data['ExterQual'].map({'Ex':100, 'Gd':90, 'TA':80, 'Fa':70, 'Po':60})
tocheck_features.drop('ExterQual', inplace = True) 
'''
ExterCond: Evaluates the present condition of the material on the exterior		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical……
       ordered categorical feature, change to scores.
'''
print('nums of na: ', all_data['ExterCond'].isnull().sum())
all_data['ExterCond'] = all_data['ExterCond'].map({'Ex':100, 'Gd':90, 'TA':80, 'Fa':70, 'Po':60})
tocheck_features.drop('ExterCond', inplace = True)
'''
Foundation: Type of foundation		
       BrkTil	Brick & Tile
       CBlock	Cinder Block
       PConc	Poured Contrete……
'''       
print('nums of na: ', all_data['Foundation'].isnull().sum())
tocheck_features.drop('Foundation', inplace = True)
'''
BsmtQual: Evaluates the height of the basement
       Ex	Excellent (100+ inches)	
       Gd	Good (90-99 inches)
       TA	Typical (80-89 inches)……
       ordered categorical feature, change to scores. 81 values missing, don't have basement, fill with 0.
'''
print('nums of na: ', all_data['BsmtQual'].isnull().sum())
all_data['BsmtQual'] = all_data['BsmtQual'].map({'Ex':100, 'Gd':90, 'TA':80, 'Fa':70, 'Po':60})
all_data['BsmtQual'] = all_data['BsmtQual'].fillna(0)
tocheck_features.drop('BsmtQual', inplace = True)       
'''
BsmtCond: Evaluates the general condition of the basement
       Ex	Excellent
       Gd	Good
       TA	Typical - slight dampness allowed……
       ordered categorical feature, change to scores. 82 values missing, don't have basement, fill with 0.
'''
print('nums of na: ', all_data['BsmtCond'].isnull().sum())
all_data['BsmtCond'] = all_data['BsmtCond'].map({'Ex':100, 'Gd':90, 'TA':80, 'Fa':70, 'Po':60})
all_data['BsmtCond'] = all_data['BsmtCond'].fillna(0)
tocheck_features.drop('BsmtCond', inplace = True) 
'''
BsmtExposure: Refers to walkout or garden level walls
       Gd	Good Exposure
       Av	Average Exposure (split levels or foyers typically score average or above)	
       Mn	Mimimum Exposure……
       82 values missing, don't have basement, fill with Nb.
'''
print('nums of na: ', all_data['BsmtExposure'].isnull().sum())
all_data['BsmtExposure'] = all_data['BsmtExposure'].fillna('Nb')
tocheck_features.drop('BsmtExposure', inplace = True) 
'''
BsmtFinType1: Rating of basement finished area
       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters……
       ordered categorical feature, change to scores. 79 values missing, don't have basement, fill with 0.
'''
print('nums of na: ', all_data['BsmtFinType1'].isnull().sum())
all_data['BsmtFinType1'] = all_data['BsmtFinType1'].map({'GLQ':100, 'ALQ':90, 'BLQ':80, 'Rec':70, 'LwQ':60, 'Unf':50})
all_data['BsmtFinType1'] = all_data['BsmtFinType1'].fillna(0)
tocheck_features.drop('BsmtFinType1', inplace = True)
'''
BsmtFinSF1: Type 1 finished square feet
      1 value missing, fill with 0.
'''
print('nums of na: ', all_data['BsmtFinSF1'].isnull().sum())
all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].fillna(0)
tocheck_features.drop('BsmtFinSF1', inplace = True) 
'''
BsmtFinType2: Rating of basement finished area (if multiple types)
       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters……
       ordered categorical feature, change to scores. 80 values missing, don't have basement, fill with 0.
'''
print('nums of na: ', all_data['BsmtFinType2'].isnull().sum())
all_data['BsmtFinType2'] = all_data['BsmtFinType2'].map({'GLQ':100, 'ALQ':90, 'BLQ':80, 'Rec':70, 'LwQ':60, 'Unf':50})
all_data['BsmtFinType2'] = all_data['BsmtFinType2'].fillna(0)
tocheck_features.drop('BsmtFinType2', inplace = True)
'''
BsmtFinSF2: Type 2 finished square feet
      1 value missing, fill with 0.
'''
print('nums of na: ', all_data['BsmtFinSF2'].isnull().sum())
all_data['BsmtFinSF2'] = all_data['BsmtFinSF2'].fillna(0)
tocheck_features.drop('BsmtFinSF2', inplace = True)
'''
BsmtUnfSF: Unfinished square feet of basement area
      1 value missing, fill with 0.
'''
print('nums of na: ', all_data['BsmtUnfSF'].isnull().sum())
all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].fillna(0)
tocheck_features.drop('BsmtUnfSF', inplace = True)
'''
TotalBsmtSF: Total square feet of basement area
      1 value missing, fill with 0.
'''
print('nums of na: ', all_data['TotalBsmtSF'].isnull().sum())
all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(0)
tocheck_features.drop('TotalBsmtSF', inplace = True)
'''
Heating: Type of heating	
       Floor	Floor Furnace
       GasA	Gas forced warm air furnace
       GasW	Gas hot water or steam heat……
'''
print('nums of na: ', all_data['Heating'].isnull().sum())
tocheck_features.drop('Heating', inplace = True)
'''
HeatingQC: Heating quality and condition
       Ex	Excellent
       Gd	Good……
'''
print('nums of na: ', all_data['HeatingQC'].isnull().sum())
all_data['HeatingQC'] = all_data['HeatingQC'].map({'Ex':100, 'Gd':90, 'TA':80, 'Fa':70, 'Po':60})
tocheck_features.drop('HeatingQC', inplace = True)
'''
CentralAir: Central air conditioning
       N	No
       Y	Yes
'''
print('nums of na: ', all_data['CentralAir'].isnull().sum())
tocheck_features.drop('CentralAir', inplace = True)
'''
Electrical: Electrical system
       SBrkr	Standard Circuit Breakers & Romex
       FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
       FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
       1 value missing, fill with most common value.
'''
print('nums of na: ', all_data['Electrical'].isnull().sum())
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
tocheck_features.drop('Electrical', inplace = True)
'''
1stFlrSF: First Floor square feet
2ndFlrSF: Second floor square feet
LowQualFinSF: Low quality finished square feet (all floors)
GrLivArea: Above grade (ground) living area square feet
'''
print('nums of na: ', all_data['1stFlrSF'].isnull().sum())
tocheck_features.drop('1stFlrSF', inplace = True)
print('nums of na: ', all_data['2ndFlrSF'].isnull().sum())
tocheck_features.drop('2ndFlrSF', inplace = True)
print('nums of na: ', all_data['LowQualFinSF'].isnull().sum())
tocheck_features.drop('LowQualFinSF', inplace = True)
print('nums of na: ', all_data['GrLivArea'].isnull().sum())
tocheck_features.drop('GrLivArea', inplace = True)
'''
BsmtFullBath: Basement full bathrooms
      2 values missing, fill with 0.
BsmtHalfBath: Basement half bathrooms
      2 values missing, fill with 0.
'''
print('nums of na: ', all_data['BsmtFullBath'].isnull().sum())
all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna(0)
tocheck_features.drop('BsmtFullBath', inplace = True)
print('nums of na: ', all_data['BsmtHalfBath'].isnull().sum())
all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].fillna(0)
tocheck_features.drop('BsmtHalfBath', inplace = True)
'''
FullBath: Full bathrooms above grade
HalfBath: Half baths above grade
BedroomAbvGr: Bedrooms above grade (does NOT include basement bedrooms)
KitchenAbvGr: Kitchens above grade
'''
print('nums of na: ', all_data['FullBath'].isnull().sum())
tocheck_features.drop('FullBath', inplace = True)
print('nums of na: ', all_data['HalfBath'].isnull().sum())
tocheck_features.drop('HalfBath', inplace = True)
print('nums of na: ', all_data['BedroomAbvGr'].isnull().sum())
tocheck_features.drop('BedroomAbvGr', inplace = True)
print('nums of na: ', all_data['KitchenAbvGr'].isnull().sum())
tocheck_features.drop('KitchenAbvGr', inplace = True)
'''
KitchenQual: Kitchen quality
       Ex	Excellent
       Gd	Good……
       1 value missing, fill with 0.
'''
print('nums of na: ', all_data['KitchenQual'].isnull().sum())
all_data['KitchenQual'] = all_data['KitchenQual'].map({'Ex':100, 'Gd':90, 'TA':80, 'Fa':70, 'Po':60})
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(0)
tocheck_features.drop('KitchenQual', inplace = True)
'''
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
'''
print('nums of na: ', all_data['TotRmsAbvGrd'].isnull().sum())
tocheck_features.drop('TotRmsAbvGrd', inplace = True)
'''
Functional: Home functionality (Assume typical unless deductions are warranted)
       Typ	Typical Functionality
       Min1	Minor Deductions 1……
       2 value missings, fill with most common value.
'''
print('nums of na: ', all_data['Functional'].isnull().sum())
all_data['Functional'] = all_data['Functional'].fillna(all_data['Functional'].mode()[0])
tocheck_features.drop('Functional', inplace = True)
'''
Fireplaces: Number of fireplaces
'''
print('nums of na: ', all_data['Fireplaces'].isnull().sum())
tocheck_features.drop('Fireplaces', inplace = True)
'''
FireplaceQu: Fireplace quality
       Ex	Excellent - Exceptional Masonry Fireplace
       Gd	Good - Masonry Fireplace in main level……
'''
print('nums of na: ', all_data['FireplaceQu'].isnull().sum())
all_data['FireplaceQu'] = all_data['FireplaceQu'].map({'Ex':100, 'Gd':90, 'TA':80, 'Fa':70, 'Po':60})
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna(0)
tocheck_features.drop('FireplaceQu', inplace = True)
'''
GarageType: Garage location		
       2Types	More than one type of garage
       Attchd	Attached to home……
       157 values missing, fill with NotExist.
'''
print('nums of na: ', all_data['GarageType'].isnull().sum())
all_data['GarageType'] = all_data['GarageType'].fillna('NotExist')
tocheck_features.drop('GarageType', inplace = True)
'''
GarageYrBlt: Year garage was built
      159 values missing, fill with 0.
'''
print('nums of na: ', all_data['GarageYrBlt'].isnull().sum())
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(0)
tocheck_features.drop('GarageYrBlt', inplace = True)
'''
GarageFinish: Interior finish of the garage
       Fin	Finished
       RFn	Rough Finished……
       159 values missing, fill with NotExist.
'''
print('nums of na: ', all_data['GarageFinish'].isnull().sum())
all_data['GarageFinish'] = all_data['GarageFinish'].fillna('NotExist')
tocheck_features.drop('GarageFinish', inplace = True)
'''
GarageCars: Size of garage in car capacity
      1 value missing, fill with 0.
GarageArea: Size of garage in square feet
'''
print('nums of na: ', all_data['GarageCars'].isnull().sum())
all_data['GarageCars'] = all_data['GarageCars'].fillna(0)
tocheck_features.drop('GarageCars', inplace = True)
print('nums of na: ', all_data['GarageArea'].isnull().sum())
all_data['GarageArea'] = all_data['GarageArea'].fillna(0)
tocheck_features.drop('GarageArea', inplace = True)
'''
GarageQual: Garage quality
       Ex	Excellent
       Gd	Good……
       159 values missing, fill with 0.
'''
print('nums of na: ', all_data['GarageQual'].isnull().sum())
all_data['GarageQual'] = all_data['GarageQual'].map({'Ex':100, 'Gd':90, 'TA':80, 'Fa':70, 'Po':60})
all_data['GarageQual'] = all_data['GarageQual'].fillna(0)
tocheck_features.drop('GarageQual', inplace = True)
'''
GarageCond: Garage condition
       Ex	Excellent
       Gd	Good……
       159 values missing, fill with 0.
'''
print('nums of na: ', all_data['GarageCond'].isnull().sum())
all_data['GarageCond'] = all_data['GarageCond'].map({'Ex':100, 'Gd':90, 'TA':80, 'Fa':70, 'Po':60})
all_data['GarageCond'] = all_data['GarageCond'].fillna(0)
tocheck_features.drop('GarageCond', inplace = True)
'''
PavedDrive: Paved driveway
       Y	Paved 
       P	Partial Pavement
       N	Dirt/Gravel
'''
print('nums of na: ', all_data['PavedDrive'].isnull().sum())
tocheck_features.drop('PavedDrive', inplace = True)
'''
WoodDeckSF: Wood deck area in square feet
OpenPorchSF: Open porch area in square feet
EnclosedPorch: Enclosed porch area in square feet
3SsnPorch: Three season porch area in square feet
ScreenPorch: Screen porch area in square feet
PoolArea: Pool area in square feet
'''
print('nums of na: ', all_data['WoodDeckSF'].isnull().sum())
tocheck_features.drop('WoodDeckSF', inplace = True)
print('nums of na: ', all_data['OpenPorchSF'].isnull().sum())
tocheck_features.drop('OpenPorchSF', inplace = True)
print('nums of na: ', all_data['EnclosedPorch'].isnull().sum())
tocheck_features.drop('EnclosedPorch', inplace = True)
print('nums of na: ', all_data['3SsnPorch'].isnull().sum())
tocheck_features.drop('3SsnPorch', inplace = True)
print('nums of na: ', all_data['ScreenPorch'].isnull().sum())
tocheck_features.drop('ScreenPorch', inplace = True)
print('nums of na: ', all_data['PoolArea'].isnull().sum())
tocheck_features.drop('PoolArea', inplace = True)
'''
PoolQC: Pool quality		
       Ex	Excellent
       Gd	Good……
       change to scores, 2909 values missing, not exist, fill with 0.
'''
print('nums of na: ', all_data['PoolQC'].isnull().sum())
all_data['PoolQC'] = all_data['PoolQC'].map({'Ex':100, 'Gd':90, 'TA':80, 'Fa':70, 'Po':60})
all_data['PoolQC'] = all_data['PoolQC'].fillna(0)
tocheck_features.drop('PoolQC', inplace = True)
'''
Fence: Fence quality		
       GdPrv	Good Privacy
       MnPrv	Minimum Privacy……
        2348 values missing, not exist, fill with NotExist.
'''
print('nums of na: ', all_data['Fence'].isnull().sum())
all_data['Fence'] = all_data['Fence'].fillna('NotExist')
tocheck_features.drop('Fence', inplace = True)
'''
MiscFeature: Miscellaneous feature not covered in other categories
       Elev	Elevator
       Gar2	2nd Garage (if not described in garage section)……
'''
print('nums of na: ', all_data['MiscFeature'].isnull().sum())
all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')
tocheck_features.drop('MiscFeature', inplace = True)
'''
MiscVal: $Value of miscellaneous feature
'''
print('nums of na: ', all_data['MiscVal'].isnull().sum())
tocheck_features.drop('MiscVal', inplace = True)
'''
MoSold: Month Sold (MM)
YrSold: Year Sold (YYYY)
'''
print('nums of na: ', all_data['MoSold'].isnull().sum())
tocheck_features.drop('MoSold', inplace = True)
print('nums of na: ', all_data['YrSold'].isnull().sum())
tocheck_features.drop('YrSold', inplace = True)
'''
SaleType: Type of sale	
       WD 	Warranty Deed - Conventional
       CWD	Warranty Deed - Cash……
       1 value missing, fill with most common value.
'''
print('nums of na: ', all_data['SaleType'].isnull().sum())
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
tocheck_features.drop('SaleType', inplace = True)
'''
SaleCondition: Condition of sale
       Normal	Normal Sale
       Abnorml	Abnormal Sale -  trade, foreclosure, short sale……
'''
print('nums of na: ', all_data['SaleCondition'].isnull().sum())
tocheck_features.drop('SaleCondition', inplace = True)

# check if there still exist feature to check
print(tocheck_features)

# check the skew of all numerical features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness = skewness[abs(skewness.Skew) > 0.75]
from scipy.special import boxcox1p
skewed_features = skewness.index
# λ=0 then equivalent to log1p
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)   
#all_data[skewed_features] = np.log1p(all_data[skewed_features])

# one-hot categorical features
all_data = pd.get_dummies(all_data)

# split train test
train = all_data[:ntrain]
test = all_data[ntrain:]

##############
## modeling ##
##############
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

# searching for good models
# Lasso
#search_alpha=[1, 0.1, 0.001, 0.0005]
alpha=[round(i, 4) for i in list(np.linspace(0.0005,0.1,20))]
lasso_scores = [1.0]*len(alpha)
for i in range(len(alpha)):
      lasso = make_pipeline(RobustScaler(), Lasso(alpha=alpha[i], random_state=13))
      rmse= np.sqrt(-cross_val_score(lasso, train.values, y_train, scoring="neg_mean_squared_error", cv = 5))
      lasso_scores[i] = (round(rmse.mean(), 4))
plt.plot(alpha, lasso_scores)
plt.title('Lasso Mean Score')
plt.show()      

# RandomForest
max_features = [round(i, 4) for i in list(np.linspace(0.1,0.99,20))]
RF_scores = [1.0]*len(max_features)
for i in range(len(max_features)):
      RF = RandomForestRegressor(max_features=max_features[i], random_state=13)
      rmse= np.sqrt(-cross_val_score(RF, train.values, y_train, scoring="neg_mean_squared_error", cv = 5))
      RF_scores[i] = (round(rmse.mean(), 4))
plt.plot(max_features, RF_scores)
plt.title('RF Mean Score')
plt.show() 

# training
lasso = Lasso(alpha = alpha[lasso_scores.index(min(lasso_scores))])
lasso.fit(train.values, y_train)
rf = RandomForestRegressor(max_features = max_features[RF_scores.index(min(RF_scores))])
rf.fit(train.values,y_train)

# predicting
y_lasso = np.expm1(lasso.predict(test))
y_rf = np.expm1(rf.predict(test))

# ensemble prediction
ensemble = (y_lasso/min(lasso_scores) + y_rf/min(RF_scores)) / (1/min(lasso_scores)+1/min(RF_scores))

################
## submission ##
################
sub = pd.DataFrame()
sub['Id'] = test_id
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)
{% endcodeblock %}