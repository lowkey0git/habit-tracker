import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pickle


def machine():
    file = pd.read_csv("companywork/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df = pd.DataFrame(file)
    drop = df.dropna()
    df_encoded = pd.get_dummies(df,columns=["Attrition", "BusinessTravel", "Department", "EducationField", "Gender"
        ,"JobRole", "MaritalStatus", "Over18", "OverTime"])
    encoded_dataframe = pd.DataFrame(df_encoded)
    x = encoded_dataframe.drop( ["Attrition_No", "Attrition_Yes"],axis=1)
    y = encoded_dataframe[["Attrition_No", "Attrition_Yes"]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    fitness = model.fit(x_train,y_train)
    y_preds = model.predict(x_test)
    print(y_preds)
machine()
