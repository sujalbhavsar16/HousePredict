import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

train_org = pd.read_csv('train.csv',engine='python')
test_org = pd.read_csv('test.csv')
# train_org.columns, test_org.columns
# print((train_org['MSZoning']=='RP').any())

# train_org.isnull().sum().to_csv('missinginfo.csv')


train_copy=train_org.copy()
test_copy=test_org.copy()

train=train_copy.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1)
test=test_copy.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1)

train_new=DataFrameImputer().fit_transform(train)
test_new=DataFrameImputer().fit_transform(test)

#
# print(train_new['MasVnrType'].isnull().any())

# cols_with_missing = [col for col in train.columns
#                                  if train[col].isnull().any()]
# print(cols_with_missing)
# my_imputer = SimpleImputer()
# # train_with_imputed_values = my_imputer.fit_transform(train[['LotFrontage','MasVnrArea']])
# train_with_imputed_values = my_imputer.fit_transform(train)
# print(train_with_imputed_values)

# print(train_with_imputed_values.isnull().sum())

# plt.figure(figsize=[300,300])
# sns.heatmap(train_org.corr(),annot=True)
# plt.show()

# +

# reg=ols('SalePrice~  C(MSSubClass)',data=train_org)
# reg=ols('SalePrice~  MSZoning',data=train_org)
# reg=ols('SalePrice~ C(MSSubClass)+MSZoning+LotFrontage+LotArea+Street \
# +Alley+LotShape+LandContour+Utilities+LotConfig+LandSlope+Neighborhood \
# +Condition1+Condition2+BldgType+HouseStyle+C(OverallQual)+C(OverallCond)+C(YearBuilt) \
# +C(YearRemodAdd)+RoofStyle+RoofMatl+Exterior1st+Exterior2nd+MasVnrType+MasVnrArea+ExterQual+ExterCond \
# +Foundation+BsmtQual+BsmtCond+BsmtExposure+BsmtFinType1+BsmtFinSF1+BsmtFinType2+BsmtFinSF2+BsmtUnfSF \
# +TotalBsmtSF+Heating+HeatingQC+CentralAir+Electrical+1stFlrSF+2ndFlrSF+LowQualFinSF+GrLivArea \
# +C(BsmtFullBath)+C(BsmtHalfBath)+C(FullBath)+C(HalfBath)+C(BedroomAbvGr)+C(KitchenAbvGr)+KitchenQual \
# +C(TotRmsAbvGrd)+Functional+C(Fireplaces)+FireplaceQu+GarageType+C(GarageYrBlt)+GarageFinish+C(GarageCars) \
# +GarageArea+GarageQual+GarageCond+PavedDrive+WoodDeckSF+OpenPorchSF+EnclosedPorch+3SsnPorch+ScreenPorch+PoolArea+PoolQC+Fence+MiscFeature+MiscVal \
# +C(MoSold)+C(YrSold)+SaleType+SaleCondition',data=train_org)

# reg=ols('SalePrice~ C(MSSubClass)+MSZoning+LotFrontage+LotArea+Street+Alley+LotShape+LandContour+Utilities+LotConfig+LandSlope+Neighborhood+Condition1+Condition2+BldgType+HouseStyle+C(OverallQual)+C(OverallCond)+C(YearBuilt)+C(YearRemodAdd)+RoofStyle+RoofMatl+Exterior1st+Exterior2nd+MasVnrType+MasVnrArea+ExterQual+ExterCond',data=train_org)
# reg_result=reg.fit()
# print(reg_result.summary())
