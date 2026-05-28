import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib as jb
from sklearn.preprocessing import OrdinalEncoder

df_car = pd.read_csv("data/indian_used_cars_raw_57480.csv")
if 'Unnamed: 0' in df_car.columns:
    df_car = df_car.drop(columns=['Unnamed: 0'])

X_car = df_car.drop(columns=['price'])
y_car = df_car['price']

cat_cols_car = X_car.select_dtypes(include=['object']).columns.tolist()
encoder_car = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
if cat_cols_car:
    X_car[cat_cols_car] = encoder_car.fit_transform(X_car[cat_cols_car].astype(str))

X_train_car, X_test_car, y_train_car, y_test_car = train_test_split(X_car, y_car, test_size=0.2, random_state=42)
model_car = RandomForestRegressor(n_estimators=100, random_state=42)
model_car.fit(X_train_car, y_train_car)

jb.dump(model_car, "model/car_price_model.joblib")
jb.dump(encoder_car, "model/car_encoder.joblib")
print("Car model and encoder trained and saved.")

df_bike = pd.read_csv("data/Used_Bikes.csv")
X_bike = df_bike.drop(columns=['price'])
y_bike = df_bike['price']

cat_cols_bike = X_bike.select_dtypes(include=['object']).columns.tolist()
encoder_bike = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
if cat_cols_bike:
    X_bike[cat_cols_bike] = encoder_bike.fit_transform(X_bike[cat_cols_bike].astype(str))

X_train_bike, X_test_bike, y_train_bike, y_test_bike = train_test_split(X_bike, y_bike, test_size=0.2, random_state=42)
model_bike = RandomForestRegressor(n_estimators=100, random_state=42)
model_bike.fit(X_train_bike, y_train_bike)

jb.dump(model_bike, "model/bike_price_model.joblib")
jb.dump(encoder_bike, "model/bike_encoder.joblib")
print("Bike model and encoder trained and saved.")
