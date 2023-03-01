import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator,TransformerMixin
from PIL import Image

HOUSING_PATH = os.path.join("datasets", "housing")

my_model = joblib.load("my_model.pkl")

rooms_ix,bedrooms_ix,population_ix,households_ix=3,4,5,6

class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
  def __init__(self,add_bedrooms_per_room=True):
    self.add_bedrooms_per_room=add_bedrooms_per_room
  def fit(self,X,y=None):
    return self
  def transform(self,X):
    rooms_per_household = X[:,rooms_ix]/X[:,households_ix]
    population_per_household=X[:,population_ix]/X[:,households_ix]
    if self.add_bedrooms_per_room:
      bedrooms_per_room=X[:,bedrooms_ix]/X[:,rooms_ix]
      return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
    else:
      return np.c_[X,rooms_per_household,population_per_household]

image = Image.open('background.jpg')

st.title("Houses Median Value Predictor Model")
st.write("""With the following parameters, you can predict 
the median value of a house based on the California Housing 
Prices""")


with st.form("my_form"):

    longitude = st.text_input(label="Input longitude", value=0)
    latitude = st.text_input(label="Input latitude", value=0)
    housing_median_age = st.text_input(label="Input house median age", value = 0)
    total_rooms = st.text_input(label="Input total rooms", value=0)
    total_bedrooms = st.text_input(label="Input total bedrooms", value = 0)
    population = st.text_input(label="Input population", value=0)
    households = st.text_input(label="Input households", value=0)
    median_income = st.text_input(label="Input median income", value=0)
   
    ocean_proximity = st.selectbox(label="Select Ocean Proximity", options=("<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"))

    submitted = st.form_submit_button("Predict the house median value")

    if(submitted):
        housing = pd.read_csv(os.path.join(HOUSING_PATH, "housing.csv")).drop("median_house_value", axis=1)
        housing_num=housing.drop("ocean_proximity",axis = 1)

        num_attribs= list(housing_num)
        cat_attribs=["ocean_proximity"]

        num_pipeline = Pipeline([
            ('imputer',SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler',StandardScaler())
        ])

        # housing_num_tr=num_pipeline.fit_transform(housing_num)

        full_pipeline=ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(),cat_attribs),])

        housing_prepared = full_pipeline.fit_transform(housing)

        data = {'longitude': [longitude],
                    'latitude': [latitude],
                    'housing_median_age':[housing_median_age],
                    'total_rooms': [total_rooms],
                    'total_bedrooms': [total_bedrooms],
                    'population': [population],
                    'households': [households],
                    'median_income': [median_income],
                    'ocean_proximity' : [ocean_proximity] }

        dt_sample = pd.DataFrame(data)

        sample_tr = full_pipeline.transform(dt_sample)
        prediction = my_model.predict(sample_tr)
        
        estring = "The median value for this house is: ${:.4f}".format(prediction[0])
        st.title(estring)

st.image(image, caption='Beautiful California')
