import joblib

model = joblib.load("model/model.pkl")

marketing = 60000
price = 25
month = 8

prediction = model.predict([[marketing, price, month]])

print("Predicted revenue:", prediction[0])
