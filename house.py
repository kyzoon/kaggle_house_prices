#!/usr/bin/evn python
# -*- coding: utf-8 -*-

__auther__ = 'preston.zhu'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.linear_model import LinearRegression

import pdb

def set_ch():
	from pylab import mpl
	mpl.rcParams['font.sans-serif'] = ['SimHei']
	mpl.rcParams['axes.unicode_minus'] = False

def plotMiscFeature(train):
	plt.subplot(141)
	train.SalePrice[train.MiscFeature.notnull()].plot.hist(grid=True)
	plt.title("MiscFeature Not Null")

	plt.subplot(142)
	train.SalePrice[train.MiscFeature.isnull()].plot.hist(grid=True)
	plt.title("MiscFeature is Null")

	plt.subplot(143)
	train.SalePrice[train.MiscFeature.notnull()].plot.hist(grid=True)
	plt.title("MiscFeature Not Null")

	plt.subplot(144, sharey=plt.gca())
	train.SalePrice[train.MiscFeature.isnull()].plot.hist(grid=True)
	plt.title("MiscFeature is Null")

	plt.show()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
						train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
	train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
														train_sizes=train_sizes, verbose=verbose)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	if plot:
		set_ch()
		plt.figure()
		plt.title(title)
		if ylim is not None:
			plt.ylim(*ylim)
		plt.xlabel(u"训练样本数")
		plt.ylabel(u"得分")
		plt.gca().invert_yaxis()
		plt.grid()

		plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
						 alpha=0.1, color="b")
		plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
						 alpha=0.1, color="r")
		plt.plot(train_sizes, train_scores_mean, 'o-', color='b', label=u"训练集上得分")
		plt.plot(train_sizes, test_scores_mean, 'o-', color='r', label=u"交叉验证集上得分")
		plt.legend(loc='best')
		plt.draw()
		plt.show()
		plt.gca().invert_yaxis()

	midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
	diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
	return midpoint, diff

def featurestudy():
	train = pd.read_csv("input/train.csv")
	test = pd.read_csv("input/test.csv")

	set_ch()

	# MSSubClass = [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190]
	# for i in range(len(MSSubClass)):
	# 	train.SalePrice[train.MSSubClass == MSSubClass[i]].plot.kde()
	# plt.axis([0, 1000000, 0, 0.00001])
	# plt.show()

	plotMiscFeature(train)

def feature():
	train = pd.read_csv("input/train.csv")
	test = pd.read_csv("input/test.csv")

	# 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
	# 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
	# 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
	# 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
	# 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
	# 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
	# 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
	# 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
	# 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
	# 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 
	# 'Bedroom', 'Kitchen', 'KitchenQual', 'TotRmsAbvGrd', 'Functional',
	# 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish',
	# 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',
	# 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
	# 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 
	# 'MoSold', 'YrSold', 'SaleType', 'SaleCondition'

	# ExtraTreesRegressor fitting loss value
	# 'LotFrontage', 
	# 'FireplaceQu', 
	# 'Fence', 'MiscFeature'

	# Nothing to do
	# OverallQual, OverallCond, BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr,
	# KitchenAbvGr, TotRmsAbvGrd, Fireplaces, GarageCars

	# Feature Scale
	# LotArea, YearBuilt, YearRemodAdd, MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF,
	# TotalBsmtSF, 1stFlrSF, 2ndFlrSF, LowQualFinSF, GrLivArea, GarageYrBlt, GarageArea
	# WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, MiscVal

	# drop Alley, PoolArea, PoolQC

 	classers = [
		'MSSubClassCat', 'MSZoningCat', 'StreetCat', 'LotShapeCat', 'LandContourCat',
		'UtilitiesCat', 'LotConfigCat', 'LandSlopeCat', 'NeighborhoodCat', 'Condition1Cat',
		'Condition2Cat', 'BldgTypeCat', 'HouseStyleCat', 'RoofStyleCat', 'RoofMatlCat',
		'Exterior1stCat', 'Exterior2ndCat', 'MasVnrTypeCat', 'ExterQualCat', 'ExterCondCat', 
		'FoundationCat', 'BsmtQualCat' , 'BsmtCondCat', 'BsmtExposureCat', 'BsmtFinType1Cat',
		'BsmtFinType2Cat', 'HeatingCat', 'HeatingQCCat', 'CentralAirCat', 'ElectricalCat',
		'KitchenQualCat', 'FunctinalCat', 'GarageTypeCat', 'GarageFinishCat', 'GarageQualCat',
		'GarageCondCat', 'PavedDriveCat', 'MoSoldCat', 'YrSoldCat', 'SaleTypeCat',
		'SaleConditionCat',

		'OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
		'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
		'GarageCars', 

		'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
		'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
		'LowQualFinSF', 'GrLivArea', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF',
		'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal'
	]

	m_train = train.shape[0]
	all_data = pd.concat([train, test], axis=0)

	all_data['MSSubClassCat'] = pd.Categorical(all_data.MSSubClass).codes
	all_data['MSZoningCat'] = pd.Categorical(all_data.MSZoning).codes
	all_data['StreetCat'] = pd.Categorical(all_data.Street).codes
	all_data['LotShapeCat'] = pd.Categorical(all_data.LotShape).codes
	all_data['LandContourCat'] = pd.Categorical(all_data.LandContour).codes

	all_data.Utilities.fillna('AllPub', inplace=True)
	all_data['UtilitiesCat'] = pd.Categorical(all_data.Utilities).codes
	all_data['LotConfigCat'] = pd.Categorical(all_data.LotConfig).codes
	all_data['LandSlopeCat'] = pd.Categorical(all_data.LandSlope).codes
	all_data['NeighborhoodCat'] = pd.Categorical(all_data.Neighborhood).codes
	all_data['Condition1Cat'] = pd.Categorical(all_data.Condition1).codes
	all_data['Condition2Cat'] = pd.Categorical(all_data.Condition2).codes
	all_data['BldgTypeCat'] = pd.Categorical(all_data.BldgType).codes
	all_data['HouseStyleCat'] = pd.Categorical(all_data.HouseStyle).codes
	all_data['RoofStyleCat'] = pd.Categorical(all_data.RoofStyle).codes
	all_data['RoofMatlCat'] = pd.Categorical(all_data.RoofMatl).codes

	all_data.Exterior1st.fillna('VinylSd', inplace=True)
	all_data['Exterior1stCat'] = pd.Categorical(all_data.Exterior1st).codes
	all_data.Exterior2nd.fillna('VinylSd', inplace=True)
	all_data['Exterior2ndCat'] = pd.Categorical(all_data.Exterior2nd).codes

	all_data.MasVnrType.fillna('None', inplace=True)
	all_data['MasVnrTypeCat'] = pd.Categorical(all_data.MasVnrType).codes
	all_data['ExterQualCat'] = pd.Categorical(all_data.ExterQual).codes
	all_data['ExterCondCat'] = pd.Categorical(all_data.ExterCond).codes
	all_data['FoundationCat'] = pd.Categorical(all_data.Foundation).codes

	all_data.BsmtQual.fillna('TA', inplace=True)
	all_data['BsmtQualCat'] = pd.Categorical(all_data.BsmtQual).codes
	all_data.BsmtCond.fillna('TA', inplace=True)
	all_data['BsmtCondCat'] = pd.Categorical(all_data.BsmtCond).codes
	all_data.BsmtExposure.fillna('No', inplace=True)
	all_data['BsmtExposureCat'] = pd.Categorical(all_data.BsmtExposure).codes

	all_data.BsmtFinType1.fillna('Unf', inplace=True)
	all_data['BsmtFinType1Cat'] = pd.Categorical(all_data.BsmtFinType1).codes
	all_data.BsmtFinType2.fillna('Unf', inplace=True)
	all_data['BsmtFinType2Cat'] = pd.Categorical(all_data.BsmtFinType2).codes
	all_data['HeatingCat'] = pd.Categorical(all_data.Heating).codes
	all_data['HeatingQCCat'] = pd.Categorical(all_data.HeatingQC).codes
	all_data['CentralAirCat'] = pd.Categorical(all_data.CentralAir).codes

	all_data.Electrical.fillna('SBrkr', inplace=True)
	all_data['ElectricalCat'] = pd.Categorical(all_data.Electrical).codes

	all_data.KitchenQual.fillna('TA', inplace=True)
	all_data['KitchenQualCat'] = pd.Categorical(all_data.KitchenQual).codes

	all_data.Functional.fillna('Typ', inplace=True)
	all_data['FunctinalCat'] = pd.Categorical(all_data.Functional).codes

	all_data.GarageType.fillna('Attchd', inplace=True)
	all_data['GarageTypeCat'] = pd.Categorical(all_data.GarageType).codes
	all_data.GarageFinish.fillna('Unf', inplace=True)
	all_data['GarageFinishCat'] = pd.Categorical(all_data.GarageFinish).codes
	all_data.GarageQual.fillna('TA', inplace=True)
	all_data['GarageQualCat'] = pd.Categorical(all_data.GarageQual).codes
	all_data.GarageCond.fillna('TA', inplace=True)
	all_data['GarageCondCat'] = pd.Categorical(all_data.GarageCond).codes
	all_data['PavedDriveCat'] = pd.Categorical(all_data.PavedDrive).codes

	all_data.SaleType.fillna('WD', inplace=True)
	all_data['SaleTypeCat'] = pd.Categorical(all_data.SaleType).codes
	all_data['SaleConditionCat'] = pd.Categorical(all_data.SaleCondition).codes
	all_data['MoSoldCat'] = pd.Categorical(all_data.MoSold).codes
	all_data['YrSoldCat'] = pd.Categorical(all_data.YrSold).codes

	all_data.MasVnrArea.fillna(0.0, inplace=True)	# 使用众数
	all_data.BsmtFinSF1.fillna(all_data.BsmtFinSF1.median(), inplace=True)
	all_data.BsmtFinSF2.fillna(0.0, inplace=True)	# 使用众数
	all_data.BsmtUnfSF.fillna(all_data.BsmtUnfSF.median(), inplace=True)
	all_data.TotalBsmtSF.fillna(all_data.TotalBsmtSF.median(), inplace=True)

	all_data.BsmtFullBath.fillna(0.0, inplace=True)
	all_data.BsmtHalfBath.fillna(0.0, inplace=True)
	all_data.GarageYrBlt.fillna(all_data.GarageYrBlt.median(), inplace=True)
	all_data.GarageCars.fillna(all_data.GarageCars.median(), inplace=True)
	all_data.GarageArea.fillna(all_data.GarageArea.median(), inplace=True)

	# Nothing to do
	# OverallQual, OverallCond, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, Fireplaces, GarageCars

	# Feature Scale
	# LotArea, YearBuilt, YearRemodAdd, MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF,
	# TotalBsmtSF, 1stFlrSF, 2ndFlrSF, LowQualFinSF, GrLivArea, GarageYrBlt, GarageArea
	# WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, MiscVal

	# drop Alley, PoolArea, PoolQC, 
	# FireplaceQu, Fence, MiscFeature
	all_data.drop(['Alley', 'PoolArea', 'PoolQC', 'FireplaceQu', 'Fence', 'MiscFeature'], axis=1, inplace=True)

	# ExtraTreesRegressor fitting loss value
	# 'LotFrontage', 
	eTreeReg = ExtraTreesRegressor(n_estimators=200)
	X_train = all_data.loc[all_data.LotFrontage.notnull(), classers]
	y_train = all_data.loc[all_data.LotFrontage.notnull(), ['LotFrontage']]
	X_test = all_data.loc[all_data.LotFrontage.isnull(), classers]
	eTreeReg.fit(X_train, np.ravel(y_train))
	all_data.loc[all_data.LotFrontage.isnull(), ['LotFrontage']] = eTreeReg.predict(X_test)

	all_data = pd.concat([
		all_data.drop(['SalePrice'], axis=1),
		all_data.SalePrice
	], axis=1)

	classers = [
		'MSSubClassCat', 'MSZoningCat', 'StreetCat', 'LotShapeCat', 'LandContourCat',
		'UtilitiesCat', 'LotConfigCat', 'LandSlopeCat', 'NeighborhoodCat', 'Condition1Cat',
		'Condition2Cat', 'BldgTypeCat', 'HouseStyleCat', 'RoofStyleCat', 'RoofMatlCat',
		'Exterior1stCat', 'Exterior2ndCat', 'MasVnrTypeCat', 'ExterQualCat', 'ExterCondCat', 
		'FoundationCat', 'BsmtQualCat' , 'BsmtCondCat', 'BsmtExposureCat', 'BsmtFinType1Cat',
		'BsmtFinType2Cat', 'HeatingCat', 'HeatingQCCat', 'CentralAirCat', 'ElectricalCat',
		'KitchenQualCat', 'FunctinalCat', 'GarageTypeCat', 'GarageFinishCat', 'GarageQualCat',
		'GarageCondCat', 'PavedDriveCat', 'MoSoldCat', 'YrSoldCat', 'SaleTypeCat',
		'SaleConditionCat',

		'OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
		'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
		'GarageCars', 

		'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
		'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
		'LowQualFinSF', 'GrLivArea', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF',
		'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal',
		'LotFrontage'
	]

	X_data = all_data.iloc[:m_train, :]
	X_train = X_data.loc[:, classers]
	y_data = all_data.iloc[:m_train, :]
	y_train = y_data.loc[:, 'SalePrice']
	X_t_data = all_data.iloc[m_train:, :]
	X_test = X_t_data.loc[:, classers]
	test_Id = X_t_data.Id.as_matrix()
	return X_train, y_train, X_test, test_Id
	# return all_data

def house():
	print('Prepareing Data...')
	X_train, y_train, X_test, test_Id = feature()
	# featurestudy()

	model_rfr = RnadomForestRegressor(n_estimators=300,
									  min_samples_leaf=4)
	model_rfr.fit(X_train, np.ravel(y_train))
	results = model_rfr.predict(X_test)

	submission = pd.DataFrame({
		'Id': test_Id,
		'SalePrice': results
	})
	submission.to_csv("prediction1_rfr.csv", index=False)

	fi = pd.DataFrame({
		"item": X_train.columns,
		"weight": model_rfr.feature_importances_
	})
	fi.to_csv("FeatureWeigth_rfr_p1.csv", index=False)

	cv = ShuffleSplit(n_splites=5, test_size=0.2, random_state=0)
	plot_learning_curve(model_rfr, 'Learning curve', X_train, y_train, cv=cv)

	model_lr = LinearRegression(normalize=True)
	model_lr.fit(X_train, np.ravel(y_train))
	results = model_lr.predict(X_test)

	submission = pd.DataFrame({
		'Id': test_Id,
		'SalePrice': results
	})
	submission.to_csv("prediction1_lr.csv", index=False)

	fi = pd.DataFrame({
		"item": X_train.columns,
		"weight": model_lr.coef_
	})
	fi.to_csv("FeatureWeigth_lr_p1.csv", index=False)

	cv = ShuffleSplit(n_splites=5, test_size=0.2, random_state=0)
	plot_learning_curve(model_lr, 'Learning curve', X_train, y_train, cv=cv)

	print('Done.')

if __name__ == '__main__':
	house()