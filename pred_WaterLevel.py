from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump
import json
import numpy as np
from lightgbm import LGBMRegressor
import lightgbm
print(lightgbm.__file__)

load_dotenv(dotenv_path='prd.env')

data_path1 = os.environ.get('DATA_PATH1')
prd_timestamp1 = os.environ.get('PRD_TIMESTAMP1')
prd_timestamp2 = os.environ.get('PRD_TIMESTAMP2')
prd_timestamp3 = os.environ.get('PRD_TIMESTAMP3')

range_values1 = os.environ.get('RANGE_VALUES1')
start, end = map(int, range_values1.split(','))
range_values1 = range(start, end)

# 讀取數據
data_path = data_path1
data = pd.read_csv(data_path)

# 定義 時間索引 欄位
def create_timestamp(data):
    # 將"Date"和"Time"合併成時間戳記並設置為索引
    data['Timestamp'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data = data.set_index('Timestamp')
    return data

# 原始資料加入時間索引
data = create_timestamp(data)
print (data)
data.head(5)

# 獲取列名
column_names = data.columns.tolist()

# 找到包含“Water Level”和“Rainfall”的列
water_level_columns = [col for col in column_names if 'Water Level' in col]
rainfall_columns = [col for col in column_names if 'Rainfall' in col]

# 選擇第一個符合條件的列
columns1 = water_level_columns[0] if water_level_columns else None
columns2 = rainfall_columns[0] if rainfall_columns else None

# 列名
if columns1:
    new_columns1_1 = f"{columns1}_monthly_mean"
    new_columns1_2 = f"{columns1}_distance"
    target_column_name = f"{columns1}_target_1"

print(new_columns1_1, new_columns1_2, target_column_name)

# 水位四舍五入小数第一位
data[columns1] = data[columns1].round(2)

# 列出所有要增加特徵的欄位名稱
columns_to_expand = [columns1, columns2]

# 1.產出_T-1到_T-72的特徵
for col in columns_to_expand:
    for t in range_values1:
        new_col_name = f"{col}_T-{t}"
        data[new_col_name] = data[col].shift(t)

# 2.產出每個月的平均水位
monthly_mean = data.groupby(data.index.month)[columns1].transform('mean')
data[new_columns1_1] = monthly_mean

# 3. 創建水位距平值：當前水位(T-0)和(T-1~T-72)與月平均水位的差值
data[new_columns1_2] = data[columns1] - monthly_mean

for t in range_values1:
    new_col_name = f"{columns1}_distance_T-{t}"
    data[new_col_name] = data[new_columns1_2].shift(t)

# 4.計算3、6、9、12、24、48、72小時的累積雨量
time_intervals = [3, 6, 9, 12, 24, 48, 72]

for interval in time_intervals:
    col_name = f"{columns2}_{interval}h_cumsum"
    data[col_name] = data[columns2].rolling(window=interval).sum()

# 5.增加T+1的雨量
for station in [columns2]:
    col = f"{station}"
    for t in range(1, 2):
        future_col_name = f"{col}_T+{t}"
        data[future_col_name] = data[col].shift(-t)

# 刪除包含NaN的行
data = data.dropna()

# 刪除不必要的列
data = data.drop(['Date', 'Time'], axis=1)

print(data)

# 特徵輸出
output_folder = 'o_feature_input'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
df = pd.DataFrame(data)

output_path = os.path.join(output_folder, f'{columns1}資料.csv')
df.to_csv(output_path, index=True)  # 如果不想保留索引，可以將 index 參數設為 False

# 標準化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data)
scaled_data = pd.DataFrame(scaled_features, columns=data.columns, index=data.index)

# 列出目標水位列
target_columns = [columns1]

# 增加目標列：T+1的目標水位
for target_col in target_columns:
    for t in range(1, 2):
        target = f'{target_col}_target_{t}'
        scaled_data[target] = data[target_col].shift(-t)

# 移除含有 NaN 的行
scaled_data = scaled_data.dropna()

# MAPE計算函數
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 設定隨機種子
seed_value = 42

# 分割數據集為訓練和測試集
train_size = int(len(scaled_data) * 0.8)
val_size = int(train_size * 0.1)
train_data = scaled_data[:train_size-val_size]
val_data = scaled_data[train_size-val_size:train_size]
test_data = scaled_data[train_size:]

X_train = train_data.drop([target_column_name], axis=1)
y_train = train_data[target_column_name]

X_val = val_data.drop([target_column_name], axis=1)
y_val = val_data[target_column_name]

X_test = test_data.drop([target_column_name], axis=1)
y_test = test_data[target_column_name]

# 訓練 LightGBM 模型
model = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=24,
    learning_rate=0.03366781071846638,
    n_estimators=675,
    max_bin=40,
    subsample=0.7124500581816137,
    subsample_freq=3,
    colsample_bytree=0.8784303388858438,
    min_child_samples=5,
    min_child_weight=10,
    verbose=0,
    device='gpu',
    random_state=seed_value
)
model.fit(X_train, y_train)

# 儲存模型和特徵名稱
if not os.path.exists("model"):
    os.makedirs("model")
dump(model, 'model/lgb_model.joblib')

# Save the feature names as a JSON file
with open('model/feature_names.json', 'w') as f:
    json.dump(list(X_train.columns), f)

# 定義多步預測函數
def recursive_forecast(model, X_test, y_test):
    forecast = []          # 存儲預測結果的列表
    true_values = []       # 存儲真實目標值的列表

    for i in range(len(X_test)):
        # 提取真实的目标值
        true_value = y_test.iloc[i]
        true_values.append(true_value)

        # 使用特征进行预测
        pred = model.predict(X_test.iloc[i].values.reshape(1, -1))
        forecast.append(pred[0])

    # 現在我們在循環外部計算R²和MAPE分數
    r2_score_value = r2_score(true_values, forecast)
    mape_score_value = mean_absolute_percentage_error(true_values, forecast)

    return forecast, true_values, r2_score_value, mape_score_value


# 找到全局最大值和最小值
global_min = min(y_train.min(), y_val.min(), y_test.min())
global_max = max(y_train.max(), y_val.max(), y_test.max())

# 進行多步預測並評估結果
sets = [
    ("Training Set", X_train, y_train),
    ("Validation Set", X_val, y_val),
    ("Test Set", X_test, y_test)
]
scores = []
results_dfs = []
for i, (set_name, X, y) in enumerate(sets):
    if set_name == "Training Set":
        start_timestamp = scaled_data.index[0]
    elif set_name == "Validation Set":
        start_timestamp = scaled_data.index[train_size - val_size]
    else:  
        start_timestamp = scaled_data.index[train_size]

    forecast, true_values, r2_scores, mape_scores = recursive_forecast(model, X.reset_index(drop=True), y.reset_index(drop=True))
    scores.append((set_name, r2_scores, mape_scores))

    # 真實值與預測值繪圖
    plt.figure(figsize=(10,6))
    plt.plot(true_values, label='True Values')  
    plt.plot(forecast, label='Forecast')
    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title(f'True Values and Forecast Values - {set_name} - {columns1}')
    plt.ylim(global_min, global_max)  # 設置y軸的限制
    if not os.path.exists("o_picture"):
        os.makedirs("o_picture")

    # 保存图像到 'picture' 文件夹
    plt.savefig(f'o_picture/{set_name}_{columns1}.png')
    plt.show()
    
    # 儲存結果與分數
    start_timestamp = scaled_data.index[scaled_data.index.get_loc(start_timestamp) + len(true_values) - len(forecast)] 
    results_df = pd.DataFrame({
        'Timestamp': pd.date_range(start=start_timestamp, periods=len(forecast), freq='H'),
        'True_Values': true_values,
        'Forecast': forecast
    })
    results_dfs.append(results_df)
# 保存结果到一个Excel文件
output_folder = 'o_water_level'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_excel_path = os.path.join(output_folder, f'{columns1}_results.xlsx')
with pd.ExcelWriter(output_excel_path) as writer:
    for i, (set_name, _, _) in enumerate(sets):
        results_dfs[i].to_excel(writer, sheet_name=f'Results_{set_name}', index=False)
        if set_name == "Training Set":
            start_timestamp = scaled_data.index[0]
        elif set_name == "Validation Set":
            start_timestamp = scaled_data.index[train_size - val_size]
        else:  # Assuming this is the "Test Set"
            start_timestamp = scaled_data.index[train_size]

        # Adjust start timestamp for the current set
        start_timestamp = scaled_data.index[scaled_data.index.get_loc(start_timestamp) + len(y.reset_index(drop=True)) - len(forecast)]

        results_df = pd.DataFrame({
            'Timestamp': pd.date_range(start=start_timestamp, periods=len(forecast), freq='H'),
            'True_Values': true_values,
            'Forecast': forecast
        })
        results_df.to_excel(writer, sheet_name=f'Results_{set_name}', index=False)
    
    scores_df = pd.DataFrame(scores, columns=['Set', 'R2_Scores', 'MAPE_Scores'])
    scores_df.to_excel(writer, sheet_name='Scores', index=False)
    
# 取得重要特徵名稱與重要評分
feature_importances = model.feature_importances_
sorted_indices = feature_importances.argsort()[::-1]  
top_20_indices = sorted_indices[:20]  

top_20_features = X_train.columns[top_20_indices]  
top_20_importances = feature_importances[top_20_indices]  

print("Important features:")
for i, (feature_name, score) in enumerate(zip(top_20_features, top_20_importances)):
    print(f"{i+1}. {feature_name} - {score}")

plt.figure(figsize=(12, 8))
plt.bar(range(len(top_20_features)), top_20_importances, tick_label=top_20_features)  
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.title('Top 20 Important Features (Sorted)')
plt.xticks(rotation=90)
plt.tight_layout()
if not os.path.exists("o_picture"):
    os.makedirs("o_picture")

# 保存图像到 'picture' 文件夹
plt.savefig(f'o_picture/Top 20 Important Features {columns1}.png')
plt.show()  

print("-------------------------------------------------------------------------")


def recursive_forecast(model, X_test, y_test, steps=288):
    forecast = []
    true_values = []
    r2_scores = []
    mape_scores = []

    for i in range(steps):
        true_value = y_test.iloc[i]
        true_values.append(true_value)
        
        pred = model.predict(X_test.iloc[i].values.reshape(1, -1))
        forecast.append(pred[0])

    return forecast, true_values


from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import json
from datetime import datetime

# 加载模型和特征名字
model = load('model/lgb_model.joblib')
with open('model/feature_names.json', 'r') as f:
    feature_names = json.load(f)
    
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def forecast_from_timestamp(timestamp_str, steps=288):
    timestamp = pd.to_datetime(timestamp_str)
    idx_start = data.index.get_loc(data.loc[data.index >= timestamp].index[0])
    idx_end = idx_start + steps

    # 獲取對應的X_test和y_test
    X_test = scaled_data.iloc[idx_start:idx_end].drop([target_column_name], axis=1).reset_index(drop=True)
    y_test = scaled_data.iloc[idx_start:idx_end][target_column_name].reset_index(drop=True)

    true_rainfall_values = data.iloc[idx_start:idx_end][columns2].reset_index(drop=True)

    # 進行多步預測
    forecast, true_values = recursive_forecast(model, X_test, y_test, steps)

    # 四捨五入預測值到小數點後兩位
    forecast = np.round(forecast, 2)

    # 计算累加后的真实值和预测值
    cum_true_values = np.cumsum(true_values)
    cum_forecast = np.cumsum(forecast)

    r2 = r2_score(true_values, forecast)
    mape = mean_absolute_percentage_error(true_values, forecast)

    print(f"T+1 to T+{steps} R²: {r2}, MAPE: {mape}")

    time_series = pd.date_range(start=timestamp + pd.Timedelta(hours=1), periods=steps, freq='H')
    
    results_df = pd.DataFrame({
        'Timestamp': time_series,  
        'Step': list(range(1, steps + 1)),
        'True Values': true_values,
        'Forecast': forecast
})
    results_df['Original Rainfall Values'] = true_rainfall_values
    descriptive_filename = f"{timestamp_str.replace(' ', '_').replace(':', '-')}_{target_column_name}.xlsx"

    # 指定你想要保存文件的文件夾
    output_folder = "o_water_level_event"
    
    # 檢查文件夾是否存在，如果不存在，則創建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 創建完整的文件路徑
    file_path = os.path.join(output_folder, descriptive_filename)

    # 使用完整的文件路徑來保存Excel文件
    with pd.ExcelWriter(file_path) as writer:
        results_df[['Step','Timestamp', 'True Values', 'Forecast', 'Original Rainfall Values']].to_excel(writer, sheet_name='Values', index=False)

        # 添加第二个分页，包括R²分数和MAPE分数
        r2_mape_df = pd.DataFrame({
            'Metric': ['R² Score', 'MAPE Score'],
            'Score': [r2, mape]
        })
        r2_mape_df.to_excel(writer, sheet_name='R2_MAPE_Scores', index=False)

    print(f"The results have been saved to '{file_path}'")

    # 真實值與預測值繪圖
    plt.figure(figsize=(10,6))
    plt.plot(true_values, label='True Values')
    plt.plot(forecast, label='Forecast')
    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title(f'True Values and Forecast Values (Timestamp: {timestamp_str})')
    plt.ylim(global_min, global_max)  # 設定y軸的限制
    if not os.path.exists("o_picture"):
        os.makedirs("o_picture")
    image_filename = f'{descriptive_filename.replace(".xlsx", "")}.png'
    image_path = os.path.join("o_picture", image_filename)
    plt.savefig(image_path, dpi=300)  
    plt.show()

# 使用函數進行預測
forecast_from_timestamp(prd_timestamp1)
forecast_from_timestamp(prd_timestamp2)
forecast_from_timestamp(prd_timestamp3)

# def forecast_from_timestamp(timestamp_str, steps=288):
#     timestamp = pd.to_datetime(timestamp_str)
#     idx_start = data.index.get_loc(data.loc[data.index >= timestamp].index[0])
#     idx_end = idx_start + steps

#     X_test = scaled_data.iloc[idx_start:idx_end].drop([target_column_name], axis=1).reset_index(drop=True)
#     y_test = scaled_data.iloc[idx_start:idx_end][target_column_name].reset_index(drop=True)

#     # 多步預測
#     forecast, true_values, r2_scores, mape_scores = recursive_forecast(model, X_test, y_test, steps=steps)

#     # 真實值與預測值繪圖
#     plt.figure(figsize=(10,6))
#     plt.plot(true_values, label='True Values')
#     plt.plot(forecast, label='Forecast')
#     plt.legend()
#     plt.xlabel('Timestamp')
#     plt.ylabel('Value')
#     plt.title('True Values and Forecast Values')
#     plt.ylim(global_min, global_max)  # 设定y轴的限制
#     plt.show()
    

    
#     # 獲得雨量數據
#     scaled_rainfall = scaled_data.iloc[idx_start:idx_end][columns2].values.reshape(-1,1)
    
#     # 創建副本
#     all_features_scaled = scaled_data.iloc[idx_start:idx_end].drop([target_column_name], axis=1).copy()
    
#     # 反標準化的直替代雨量的值
#     all_features_scaled[columns2] = scaled_rainfall.ravel()
    
    
#     # 反標準化
#     all_features_original = scaler.inverse_transform(all_features_scaled.values)
    
#     # 將NumPy數組轉換回DataFrame
#     all_features_original_df = pd.DataFrame(all_features_original, columns=all_features_scaled.columns)
    
#     # 獲得雨量反標準化特徵
#     filename_timestamp = timestamp.strftime('%Y%m%d_%H%M%S')
    
#     if not os.path.exists("o_picture"):
#         os.makedirs("o_picture")

#     # 獲得雨量反標準化特徵
#     original_rainfall = all_features_original_df.loc[:, columns2].values    
#     time_index = np.array(data.index)[idx_start:idx_end]
#     original_rainfall = np.ravel(original_rainfall)
#     # 繪製
#     plt.figure(figsize=(10,6))
#     plt.plot(time_index, original_rainfall, label='Rainfall')
#     plt.xlabel('Timestamp')
#     plt.ylabel('Rainfall')
#     plt.title('Rainfall Over Time')
#     plt.legend()
#     plt.savefig(f"o_picture/{filename_timestamp}_rainfall.png")
#     plt.show()

#     return forecast, true_values, r2_scores, mape_scores

# # 使用函數進行預測  
# forecast_from_timestamp(prd_timestamp1)
# forecast_from_timestamp(prd_timestamp2)
# forecast_from_timestamp(prd_timestamp3)
