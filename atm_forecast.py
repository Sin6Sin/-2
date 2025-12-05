import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

print("=== ATM CASH FORECAST ===")

df = pd.read_csv('atm_cash_management_dataset.csv')  # Скачай с Kaggle
print(f"Загружено {len(df)} строк")

df['Date'] = pd.to_datetime(df['DateTime']).dt.date
df['day_of_week'] = pd.to_datetime(df['DateTime']).dt.dayofweek
df['is_holiday'] = df.get('IsHoliday', 0).astype(int)
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

df['balance_lag1'] = df['Balance'].shift(1)
df['balance_lag7'] = df['Balance'].shift(7)
df['withdrawals_lag1'] = df['Withdrawals'].shift(1)

# Rolling
df['rolling_mean_7'] = df['Withdrawals'].rolling(7).mean()

df = df.dropna()
print("Признаки созданы")

features = ['balance_lag1', 'balance_lag7', 'withdrawals_lag1', 'rolling_mean_7', 
           'day_of_week', 'is_holiday', 'is_weekend']
X = df[features]
y = df['Balance'].shift(-1)[:-1]
X = X[:-1]

model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
print("МОДЕЛЬ ОБУЧЕНА!")

lgb.plot_importance(model, max_num_features=10, figsize=(8,6))
plt.savefig('importance.png')
plt.close()
print("Сохранено: importance.png")

# SHAP анализ
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X[:100])
shap.summary_plot(shap_values, X[:100], show=False)
plt.savefig('shap.png')
plt.close()
print("Сохранено: shap.png")

# Прогноз на 14 дней
last_data = df[features].tail(1)
forecast = []
for i in range(14):
    pred = model.predict(last_data)[0]
    forecast.append(pred)
    # Обновляем lag1 для следующего дня
    last_data['balance_lag1'] = pred

dates = pd.date_range(start=df['Date'].max() + pd.Timedelta('1D'), periods=14)
forecast_df = pd.DataFrame({'date': dates, 'balance': forecast})
forecast_df.to_csv('forecast_14days.csv', index=False)
print("\n=== ПРОГНОЗ ===")
print(forecast_df)
print("Сохранено: forecast_14days.csv")

# Сохранение модели
joblib.dump(model, 'model.pkl')
print("МОДЕЛЬ СОХРАНЕНА: model.pkl")

print("ВСЕ ГОТОВО")

