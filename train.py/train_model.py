import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import train_test_split
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("data/business_data.csv")

data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.month

X = data[['marketing', 'price', 'month']]
y = data['revenue']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print("MAE:", mae)

joblib.dump(model, "model/model.pkl")
print("Model saved.")
