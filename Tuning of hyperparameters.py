#導入
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold,cross_val_score,train_test_split,cross_val_predict,GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler 
from scipy import stats
from scipy.stats import norm,skew
import warnings
from sklearn.preprocessing import RobustScaler
warnings.filterwarnings("ignore")
%matplotlib inline

# 讀取數據
train = pd.read_csv(r"C:\Users\rex\Desktop\賴老師連續流\part1\無氣提.csv") 

print (train)
train.head(5)

test = pd.read_csv(r"C:\Users\rex\Desktop\賴老師連續流\part1\氣提.csv")

print (test)
test.head(5)

# ND轉換NA
train = train.replace("ND", pd.NA)
test = test.replace("ND" , pd.NA)

# 將百分比轉換為數值
percentage_cols = ["SCOD removal(%)", "TCOD removal(%)", "C/N ratio", "Hydrolysis(%)", "Acidogensis(%)", "Methanogenesis(%)", "VS/TS(%)", "TS removal(%)", "VS removal(%)", "CH4 concentration (%)"]
for col in percentage_cols:
    # 確保所有值為字串型態
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)
    
    # 如果該值包含數字和%，則將 "%" 字符去掉並轉換為浮點數；否則直接轉換為浮點數
    train[col] = train[col].apply(lambda x: float(x.replace("%","")) / 100.0 if "%" in x else float(x) if x.replace(".","").isdigit() else np.nan)
    test[col] = test[col].apply(lambda x: float(x.replace("%","")) / 100.0 if "%" in x else float(x) if x.replace(".","").isdigit() else np.nan)

# # 處理ND值，如果某個特徵有連續20個ND，就刪除該特徵
# features_to_drop = [column for column in train.columns if (train[column] == "ND").rolling(window=10).sum().max() >= 20]
# train = train.drop(columns=features_to_drop)

# 使用前後的值加總除以2進行插值
train = train.interpolate(method="linear")
test = test.interpolate(method="linear")


train["SCOD removal(%)"] = train["SCOD removal(%)"].fillna(0)

train = train.iloc[2:].reset_index(drop=True)
test = test.iloc[2:].reset_index(drop=True)

#對分類變量編碼
from sklearn.preprocessing import LabelEncoder
cols = ('Days', 'pH', 'Temp', 'CH4 concentration (%)', 'Infulent VS',
                     'Influent TCOD (g/L)', 'Influent SCOD(g/L)', 'SCOD(g/L)', 'TCOD(g/L)',
                     'TCOD removal(%)', 'SCOD removal(%)', 'C/N ratio', 'Butanol', 'Acetate',
                     'Propionate', 'i-Butyrate', 'n-Butyrate', 'Total VFA', 'TN', 'TAN',
                     'TKN', 'NH4+-N', 'Hydrolysis(%)', 'Acidogensis(%)', 'Methanogenesis(%)',
                     'VS Influent(g/L)', 'TS Influent(g/L)', 'TS(g/L)', 'VS(g/L)',
                     'VS/TS(%)', 'TS removal(%)', 'VS removal(%)', 'OLR (g-VS/(L· d))',
                     'MPR', 'MY', 'BPR', 'BY')

for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values)) 
    train[c] = lbl.transform(list(train[c].values))

for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(test[c].values)) 
    test[c] = lbl.transform(list(test[c].values))    
 
#查看數據維度        
print('Shape all_data: {}'.format(train.shape))
print('Shape all_data: {}'.format(test.shape))

#轉換成虛擬變數
train_dummy_data = pd.get_dummies(train)
train_dummy_data["SCOD removal(%)"] = train_dummy_data["SCOD removal(%)"].fillna(0)
train_dummy_data.head()

#數值型數據
numeric_cols = train_dummy_data.columns[train_dummy_data.dtypes !='object']
numeric_cols

#標準化:（X-X'）/S  進行數據轉換  X 為原始值，X' 為平均值，S 為標準差
numeric_cols_mean =train_dummy_data.loc[: ,numeric_cols].mean() # mean() 函式計算數值型特徵的平均值
numeric_cols_std = train_dummy_data.loc[: ,numeric_cols].std() #std() 函式計算數值型特徵的標準差
train_dummy_data.loc[: ,numeric_cols] = (train_dummy_data.loc[: ,numeric_cols] - numeric_cols_mean)/numeric_cols_std #轉換後的數值型特徵資料儲存在 train_dummy_data 變數中

#隨機森林塞選變


# 準備特徵和目標變量
features = ['Days', 'pH', 'Temp', 'CH4 concentration (%)', 'Infulent VS',
            'Influent TCOD (g/L)', 'Influent SCOD(g/L)', 'SCOD(g/L)', 'TCOD(g/L)',
            'TCOD removal(%)', 'SCOD removal(%)', 'C/N ratio', 'Butanol', 'Acetate',
            'Propionate', 'i-Butyrate', 'n-Butyrate', 'Total VFA', 'TN', 'TAN',
            'TKN', 'NH4+-N', 'Hydrolysis(%)', 'Acidogensis(%)', 'Methanogenesis(%)',
            'VS Influent(g/L)', 'TS Influent(g/L)', 'TS(g/L)', 'VS(g/L)',
            'VS/TS(%)', 'TS removal(%)', 'VS removal(%)', 'OLR (g-VS/(L· d))']

targets = ['MPR', 'MY', 'BPR', 'BY']  # 您的目標變量

# 創建特徵和目標變量的數據集
X = train_dummy_data[features]
y = train_dummy_data[targets]

# 構建隨機森林模型
model = RandomForestRegressor(n_estimators=100)

# 訓練模型
model.fit(X, y)

# 提取變數重要性
importance_scores = model.feature_importances_

# 設置閾值為分位數
threshold = np.percentile(importance_scores, 80)

# 選擇重要性分數大於閾值的特徵
selected_features = [feat for feat, score in zip(features, importance_scores) if score > threshold]


# 可視化變數重要性


plt.bar(features, importance_scores)
plt.xlabel('Features')
plt.ylabel('Importance Scores')
plt.xticks(rotation=90)
plt.show()

# 將特徵進行數學轉換
train_dummy_data['Methanogenesis_Squared'] = np.square(train_dummy_data['Methanogenesis(%)']) #平方
train_dummy_data['Methanogenesis_SquareRoot'] = np.sqrt(train_dummy_data['Methanogenesis(%)']) #平方根
train_dummy_data['Methanogenesis_Log'] = np.log(train_dummy_data['Methanogenesis(%)'] + 1)  # 自然對數，加1避免對數轉換中的0值

# 結合多個特徵來創建新的特徵
train_dummy_data['Interaction_Meth_Hydro'] = train_dummy_data['Methanogenesis(%)'] * train_dummy_data['Hydrolysis(%)'] #平方
train_dummy_data['Sum_Meth_Hydro'] = train_dummy_data['Methanogenesis(%)'] + train_dummy_data['Hydrolysis(%)'] #平方根
train_dummy_data['Ratio_Meth_Hydro'] = train_dummy_data['Methanogenesis(%)'] / (train_dummy_data['Hydrolysis(%)'] + 1)  # 加1避免除以0

train_dummy_data.fillna(0, inplace=True)

# 檢查衍生特徵是否成功添加到數據集中
print(train_dummy_data.head())

#隨機森林塞選變

# 準備特徵和目標變量
features = ['Days', 'pH', 'Temp', 'CH4 concentration (%)', 'Infulent VS',
            'Influent TCOD (g/L)', 'Influent SCOD(g/L)', 'SCOD(g/L)', 'TCOD(g/L)',
            'TCOD removal(%)', 'SCOD removal(%)', 'C/N ratio', 'Butanol', 'Acetate',
            'Propionate', 'i-Butyrate', 'n-Butyrate', 'Total VFA', 'TN', 'TAN',
            'TKN', 'NH4+-N', 'Hydrolysis(%)', 'Acidogensis(%)', 'Methanogenesis(%)',
            'VS Influent(g/L)', 'TS Influent(g/L)', 'TS(g/L)', 'VS(g/L)',
            'VS/TS(%)', 'TS removal(%)', 'VS removal(%)', 'OLR (g-VS/(L· d))',
            'Methanogenesis_Squared', 'Methanogenesis_SquareRoot', 'Methanogenesis_Log',
            'Interaction_Meth_Hydro', 'Sum_Meth_Hydro', 'Ratio_Meth_Hydro']

targets = ['MPR', 'MY', 'BPR', 'BY']  # 目標變量
target_MPR = train_dummy_data['MPR']
target_MY = train_dummy_data['MY']
target_BPR = train_dummy_data['BPR']
target_BY = train_dummy_data['BY']

for target in targets:
    print("Target:", target)
    
    # 创建特征和目标变量的数据集
    X = train_dummy_data[features]
    y = train_dummy_data[target]
    
    # 构建随机森林模型
    model = RandomForestRegressor(n_estimators=100)
    
    # 训练模型
    model.fit(X, y)
    
    # 提取变量重要性
    importance_scores = model.feature_importances_
    
    # 可视化变量重要性
    plt.bar(features, importance_scores)
    plt.xlabel('Features')
    plt.ylabel('Importance Scores')
    plt.xticks(rotation=90)
    plt.show()

#迭代特徵選擇
from sklearn.feature_selection import RFE

model = RandomForestRegressor(n_estimators=300)

rfe = RFE(estimator=model, n_features_to_select=10)  # 選擇保留的特徵數量

X_selected = rfe.fit_transform(X, y)

selected_feature_indexes = rfe.get_support(indices=True)
selected_features = [features[idx] for idx in selected_feature_indexes]



print("使用遞歸特徵消除 (RFE) 選擇特徵:")
print(selected_features)

X_selected = pd.DataFrame(X_selected, columns=selected_features)
scaler = RobustScaler()
X_selected_scaled = scaler.fit_transform(X_selected)
model.fit(X_selected_scaled, y)



param_grid_xgb = {
    'colsample_bytree': [0.4, 0.6, 0.8],
    'gamma': [0.0, 0.1],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [1000, 2000, 3000],
    'subsample': [0.5, 0.7, 0.9]
}

param_grid_lgb = {
    'num_leaves': [5, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [500, 1000, 2000],
    'max_bin': [50, 60, 70],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.2, 0.5, 0.8]
}

# 交叉驗證
n_folds = 5

# RMSE
def rmsle_cv(model, X, y):
    kf = KFold(n_folds, shuffle=True, random_state=42).split(X.values)
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))
    return rmse

# MAE
def mae_cv(model, X, y):
    kf = KFold(n_folds, shuffle=True, random_state=42).split(X.values)
    mae = -cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=kf)
    return mae

# R-squared
def r2_cv(model, X, y):
    kf = KFold(n_folds, shuffle=True, random_state=42).split(X.values)
    r2 = cross_val_score(model, X, y, scoring="r2", cv=kf)
    return r2

# 用來儲存模型的字典
models_dict = {}

#拆分訓練集和驗證集

# 定義訓練集和驗證集的比例，使用80%的數據作為訓練集，20%的數據作為驗證集
train_size = 0.8

for target_name, target in zip(['MPR', 'MY', 'BPR', 'BY'], [target_MPR, target_MY, target_BPR, target_BY]):
    print("Target:", target_name)
    
    X = train_dummy_data[features].loc[target.index]
    y_selected = target

    # 使用train_test_split來拆分訓練集和驗證集
    X_train, X_valid, y_train, y_valid = train_test_split(X, y_selected, train_size=train_size, random_state=1)

    # Lasso
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
    lasso.fit(X_train, y_train)
    
    # 在驗證集上進行預測並評估性能
    y_pred = lasso.predict(X_valid)
    
    # 更改評估函數，使之適用於驗證集
    mae = mae_cv(lasso, X_train, y_train)
    
    rmse = rmsle_cv(lasso, X_train, y_train)
    
    r2 = r2_cv(lasso, X_train, y_train)

    print("\nLasso MAE score: {:.4f} ({:.4f})\n".format(mae.mean(), mae.std()))
    
    print("\nLasso RMSE score: {:.4f} ({:.4f})\n".format(rmse.mean(), rmse.std()))
    
    print("\nLasso R^2 score: {:.4f} ({:.4f})\n".format(r2.mean(), r2.std()))
    models_dict[target_name + '_lasso'] = lasso
    
    
    #ENet
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    ENet.fit(X_train, y_train)
    mae = mae_cv(ENet, X_train, y_train)
    rmse = rmsle_cv(ENet, X_train, y_train)
    r2 = r2_cv(ENet, X_train, y_train)
    print("\nElasticNet MAE score: {:.4f} ({:.4f})\n".format(mae.mean(), mae.std()))
    print("\nElasticNet RMSE score: {:.4f} ({:.4f})\n".format(rmse.mean(), rmse.std()))
    print("\nElasticNet R^2 score: {:.4f} ({:.4f})\n".format(r2.mean(), r2.std()))
    models_dict[target_name + '_ENet'] = ENet
    
    #KRR
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree= 2,coef0=2.5)
    KRR.fit(X_train, y_train)
    mae = mae_cv(KRR, X_train, y_train)
    rmse = rmsle_cv(KRR, X_train, y_train)
    r2 = r2_cv(KRR, X_train, y_train)
    print("\nKernel Ridge RMSE score: {:.4f} ({:.4f})\n".format(mae.mean(), mae.std()))
    print("\nKernel Ridge RMSE score: {:.4f} ({:.4f})\n".format(rmse.mean(), rmse.std()))
    print("\nKernel Ridge R^2 score: {:.4f} ({:.4f})\n".format(r2.mean(), r2.std()))
    models_dict[target_name + '_KRR'] = KRR
    
    #梯度提升
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                    max_depth=4, max_features='sqrt',
                                    min_samples_leaf=15, min_samples_split=10,
                                    loss='huber', random_state=5)
    GBoost.fit(X_train, y_train)
    mae = mae_cv(GBoost, X_train, y_train)
    rmse = rmsle_cv(GBoost, X_train, y_train)
    r2 = r2_cv(GBoost, X_train, y_train)
    print("\nGradient Boosting MAE score: {:.4f} ({:.4f})\n".format(mae.mean(), mae.std()))
    print("\nGradient Boosting RMSE score: {:.4f} ({:.4f})\n".format(rmse.mean(), rmse.std()))
    print("\nGradient Boosting R^2 score: {:.4f} ({:.4f})\n".format(r2.mean(), r2.std()))
    models_dict[target_name + '_GBoost'] = GBoost
    
    
    
    #XGB
    # model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.4608,     # colsamples_bytree 是列的採樣率    # gamma 是節點劃分所需的最小損失減少值
    #                          learning_rate=0.05, max_depth=3,               # learning_rate 是每次迭代的步長大小 # max_depth 是樹的最大深度
    #                          min_child_weight=1.7817,n_estimators=2200,     # min_child_weight 是子節點所需最小樣本數   # n_estimators 是樹的數量
    #                          subsample=0.5213,                              # subsample 是行的採樣率            
    #                          reg_alpha=0.4640, reg_lambda=0.8571,           # reg_alpha 和 reg_lambda 是 L1 和 L2 正則化項的權重
    #                          random_state=7, nthread=-1)                    # nthread 是 CPU 的使用數量
    model_xgb = xgb.XGBRegressor()
    #XGBoost
    model_xgb.fit(X_train, y_train)
    mae = mae_cv(model_xgb, X_train, y_train)
    rmse = rmsle_cv(model_xgb, X_train, y_train)
    r2 = r2_cv(model_xgb, X_train, y_train)
    print("\nXgboost MAE score: {:.4f} ({:.4f})\n".format(mae.mean(), mae.std()))
    print("\nXgboost RMSE score: {:.4f} ({:.4f})\n".format(rmse.mean(), rmse.std()))
    print("\nXgboost R^2 score: {:.4f} ({:.4f})\n".format(r2.mean(), r2.std()))
    models_dict[target_name + '_xgb'] = model_xgb

    #lgb
    # model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
    #                             learning_rate=0.05, n_estimators=720,
    #                             max_bin=55, subsample=0.8, subsample_freq=5,
    #                             colsample_bytree=0.2319,
    #                             min_child_samples=6, min_child_weight=11)
    model_lgb = lgb.LGBMRegressor()
    model_lgb.fit(X_train, y_train)
    mae = mae_cv(model_lgb, X_train, y_train)
    rmse = score = rmsle_cv(model_lgb, X_train, y_train)
    r2 = r2_cv(model_lgb, X_train, y_train)
    print("\nLGB MAE score: {:.4f} ({:.4f})\n".format(mae.mean(), mae.std()))
    print("\nLGB RMSE score: {:.4f} ({:.4f})\n" .format(rmse.mean(), rmse.std()))
    print("\nLGB R^2 score: {:.4f} ({:.4f})\n".format(r2.mean(), r2.std()))
    models_dict[target_name + '_lgb'] = model_lgb