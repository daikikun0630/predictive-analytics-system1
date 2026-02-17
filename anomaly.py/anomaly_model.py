import pandas as pd
from sklearn.ensemble import IsolationForest

data = pd.read_csv("data/business_data.csv")

model = IsolationForest(contamination=0.05)
data['anomaly'] = model.fit_predict(data[['revenue']])

print(data[data['anomaly'] == -1])
