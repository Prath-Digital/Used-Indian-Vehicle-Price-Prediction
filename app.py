import streamlit as st
import pandas as pd
import joblib as jb
import datetime

st.title("Used Indian Vehicle Price Prediction")

vehicle_type = st.radio("Select Vehicle Type", ["Car", "Bike"])

def format_inr(n):
    s = str(int(n))
    if len(s) <= 3:
        return s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.append(rest)
        return ",".join(parts[::-1]) + "," + last3

if vehicle_type == "Car":
    st.subheader("Car Price Predictor")
    try:
        df_car = pd.read_csv("data/indian_used_cars_raw_57480.csv")
        X_car = df_car.drop(columns=["price"])
        if 'Unnamed: 0' in X_car.columns:
            X_car = X_car.drop(columns=['Unnamed: 0'])
            
        ml_model_car = jb.load("model/car_price_model.joblib")
        encoder_car = jb.load("model/car_encoder.joblib")
        
        user_input_car = {}
        
        brands = X_car["brand"].unique().tolist()
        brand = st.selectbox("Select brand", brands)
        user_input_car["brand"] = brand
        
        models = X_car[X_car["brand"] == brand]["model"].unique().tolist()
        car_model = st.selectbox("Select model", models)
        user_input_car["model"] = car_model
        
        bodytypes = X_car[X_car["model"] == car_model]["bodytype"].unique().tolist()
        if len(bodytypes) == 1:
            bodytype = bodytypes[0]
            st.write(f"Bodytype: {bodytype}")
        else:
            bodytype = st.selectbox("Select bodytype", bodytypes)
        user_input_car["bodytype"] = bodytype
        
        fuel_types = X_car["fuel_type"].unique().tolist()
        fuel_type = st.selectbox("Select fuel type", fuel_types)
        user_input_car["fuel_type"] = fuel_type
        
        trans_types = X_car[X_car["model"] == car_model]["transmission_type"].unique().tolist()
        trans_type = st.selectbox("Select transmission type", trans_types)
        user_input_car["transmission_type"] = trans_type
        
        cities = X_car["city"].unique().tolist()
        city = st.selectbox("Select city", cities)
        user_input_car["city"] = city
        
        current_year = datetime.datetime.now().year
        manufacturing_year = st.number_input(
            "Enter manufacturing year",
            min_value=1949,
            max_value=current_year,
            value=2018,
        )
        user_input_car["manufacturing_year"] = manufacturing_year
        
        min_owner = int(X_car["number_of_owners"].min())
        max_owner = int(X_car["number_of_owners"].max())
        number_of_owners = st.number_input(
            "Enter number of owners", min_value=min_owner, max_value=max_owner, value=min_owner
        )
        user_input_car["number_of_owners"] = number_of_owners
        
        km_driven = st.number_input("Enter km driven", value=25000)
        user_input_car["km_driven"] = km_driven
        
        if st.button("Predict Price"):
            input_df = pd.DataFrame([user_input_car])
            feature_cols = [col for col in X_car.columns if col != "Unnamed: 0"]
            input_df = input_df[feature_cols]
            
            cat_cols = X_car.select_dtypes(include=["object"]).columns.tolist()
            if cat_cols:
                input_df[cat_cols] = encoder_car.transform(input_df[cat_cols].astype(str))
                
            pred = ml_model_car.predict(input_df)[0]
            st.success(f"Estimated Car Price: ₹{format_inr(pred)}")
            
    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.subheader("Bike Price Predictor")
    try:
        df_bike = pd.read_csv("data/Used_Bikes.csv")
        X_bike = df_bike.drop(columns=["price"])
        if 'Unnamed: 0' in X_bike.columns:
            X_bike = X_bike.drop(columns=['Unnamed: 0'])
            
        ml_model_bike = jb.load("model/bike_price_model.joblib")
        encoder_bike = jb.load("model/bike_encoder.joblib")
        
        user_input_bike = {}
        
        brands_bike = X_bike["brand"].unique().tolist()
        brand_bike = st.selectbox("Select brand", sorted(brands_bike))
        user_input_bike["brand"] = brand_bike
        
        bike_names = X_bike[X_bike["brand"] == brand_bike]["bike_name"].unique().tolist()
        bike_name = st.selectbox("Select bike name", sorted(bike_names))
        user_input_bike["bike_name"] = bike_name
        
        cities_bike = X_bike["city"].unique().tolist()
        city_bike = st.selectbox("Select city", sorted(cities_bike))
        user_input_bike["city"] = city_bike
        
        owners_bike = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth Owner Or More']
        owner_bike = st.selectbox("Select owner type", owners_bike)
        user_input_bike["owner"] = owner_bike
        
        kms_driven = st.number_input("Enter kms driven", value=12000)
        user_input_bike["kms_driven"] = kms_driven
        
        bike_data_subset = X_bike[X_bike["bike_name"] == bike_name]
        default_power = float(bike_data_subset["power"].iloc[0]) if not bike_data_subset.empty else 150.0
        power = st.number_input(
            "Enter power (cc)",
            min_value=50.0,
            max_value=2000.0,
            value=default_power,
            step=10.0,
        )
        user_input_bike["power"] = power
        
        default_age = int(bike_data_subset["age"].median()) if not bike_data_subset.empty else 3
        age = st.number_input(
            "Enter age (years)",
            min_value=0,
            max_value=40,
            value=int(default_age),
        )
        user_input_bike["age"] = float(age)
        
        if st.button("Predict Price"):
            input_df_bike = pd.DataFrame([user_input_bike])
            feature_cols_bike = [col for col in X_bike.columns]
            input_df_bike = input_df_bike[feature_cols_bike]
            
            cat_cols_bike = X_bike.select_dtypes(include=["object"]).columns.tolist()
            if cat_cols_bike:
                input_df_bike[cat_cols_bike] = encoder_bike.transform(input_df_bike[cat_cols_bike].astype(str))
                
            pred = ml_model_bike.predict(input_df_bike)[0]
            st.success(f"Estimated Bike Price: ₹{format_inr(pred)}")
            
    except Exception as e:
        st.error(f"Error: {e}")
