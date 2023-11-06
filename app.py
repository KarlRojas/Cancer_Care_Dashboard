from shiny import Inputs, Outputs, Session, App, ui, render, reactive
from pathlib import Path
from test import treatment_file, data_separation, data_split, LR, evaluate_linear_regression, train_and_evaluate_random_forest, plot_linear_regression_results, create_shap_waterfall_chart, shap_beeswarm_plot, shap_violin_plot, Split_and_Shap,create_and_display_graph_test, Joblib
from test import plot_RF_results, RF, patient_data, pred_plot
import shiny as x
import requests_fhir as requests
import pandas as pd
from datetime import datetime
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import shap
import sklearn
from shinywidgets import output_widget, register_widget, render_widget
from io import BytesIO
import plotly.express as px
from asyncio import sleep
import numpy as np
import plotly.graph_objs as go
from shiny import App, reactive, ui, Session, render, Inputs, Outputs
from sklearn.linear_model import LinearRegression
import base64
import random
import shinyswatch
from htmltools import css
import seaborn as sns
from flask import send_file
import plotly.express as px
import streamlit as st
import tempfile
import os
import joblib
BASE_URL = 'https://test/fhir'

#Opening the Treatment CSV file
infile = Path(__file__).parent / "data/treat_data.csv"
treat = pd.read_csv(infile)

#Loading the pre-trained model and Opening the Simulated Data
model = joblib.load('model.joblib')
data = pd.read_csv('simulated_data.csv')

#Opening the Snomed CSV file
infiles = Path(__file__).parent / "data/snomed.csv"
snomed = pd.read_csv(infiles)

app_ui = ui.page_fluid(
    ui.panel_title("AI Dashboard for Cancer Care"), 
    {"style": "text-align : center;" "font-weight : bold;"},
    output_widget("my_widget"),
    ui.panel_main(
        shinyswatch.theme.darkly(),
        x.ui.navset_pill_list(
             ui.nav(
                "Patient Informations",
                x.ui.card(
                    x.ui.card_header("Patient Info"),
                    ui.input_numeric("patient_id", "Enter the Patient ID", 2, min=1, max=1000000000),
                    ui.p(ui.input_action_button("send", "Enter", class_="btn-primary")),
                    ui.output_table("patient_table"),
                ),
                x.ui.card(
                    x.ui.card_header("Patient History"),
                    ui.input_text("snowmed", "Snowmed code", value='chol'),
                    ui.input_numeric("patient2", "Enter the Patient ID", 2, min=1, max=1000000000),
                    ui.p(ui.input_action_button("send2", "Enter", class_="btn-primary")),
                    ui.output_table("history"),
                ),
            ),
            ui.nav(
                "Linear Regression & Random Forest",
                ui.row(
                    ui.column(
                        6,
                        x.ui.card(
                            x.ui.card_header("Linear Regression "),
                            ui.output_plot("Data_test"),
                        ),
                        x.ui.card(
                            x.ui.card_header("Linear Regression Waterfall chart"),
                            ui.output_plot("WaterfallPNG"),
                        ),
                    ),
                    ui.column(
                        6,
                        x.ui.card(
                            x.ui.card_header("Random Forest"),
                            ui.output_plot("Random_Forest_plot"),
                        ),

                        x.ui.card(
                            x.ui.card_header("Waterfall Random Forest"),
                            ui.output_plot("WaterRF")
                        ),
                    ),
                ),
            ),
            ui.nav(
                "BeeSwarm & Violin Graphs",
                x.ui.card(
                    x.ui.card_header("Positive and negative SHAP features"),
                    ui.output_text("positive_negative"),
                    x.ui.card_header("BeeSwarm"),
                    ui.output_plot("plot_bee"),
                ),
                x.ui.card(
                    x.ui.card_header("Violin Chart"),
                    ui.output_plot("plot_violin")
                ),
            ),
            ui.nav(
                "What if analysis",
                    x.ui.card(
                    x.ui.card_header("What if"),
                    ui.p(ui.input_action_button("pred", "Create a new Prediction!", class_="btn-primary")),
                    ui.input_slider(
                        "age",
                        "Age",
                        0,
                        120,
                        65,
                    ),
                    ui.input_slider(
                        "blood_pressure",
                        "Blood_Pressure",
                        60,
                        150,
                        100,
                        step=0.01,
                        animate = True
                    ),
                    ui.input_selectize(
                        "gender",
                        "Choose your gender:",
                        {
                            "Gender": {"M": "Male", "F": "Female", "O": "Others"},
                        },
                        multiple=False,
                        selected=False,
                    ),
                    ui.input_selectize(
                        "diabetes",
                        "Diabetes or not:",
                        {
                            "Diabetes": {"Y": "Yes", "N": "No"},
                        },
                        multiple=False,
                        selected=False,
                    ),
                    ui.output_text("New_Prediction"),
                ),
                x.ui.card(
                    x.ui.card_header("Current plot"),
                    ui.output_plot("Current"),
                ),
                x.ui.card(
                    x.ui.card_header("What if plot"),
                    ui.output_plot("new_LR_plt"),
                ),
            ),
            
            ui.nav(
                "Joblib Prediction",
                x.ui.card(
                    x.ui.card_header("Patient ROW"),
                    ui.input_numeric("patient_row", "Enter the Patient ROW", 1, min=1, max=len(data)),
                    ui.p(ui.input_action_button("send3", "Enter", class_="btn-primary")),
                    ui.output_text("patient_Row"),
                ),
                x.ui.card(
                    x.ui.card_header("Predictions results"),
                    ui.output_text("Pred"),
                ),
                x.ui.card(
                    x.ui.card_header("Predictions Plot"),
                    ui.output_plot("Pred_plot"),

                )
            ),
           
            ui.nav(
                "Treatment Plans",
                
            ),
            ui.nav(
                "Feedback and Support"
            )
        )
    ),
)

#Server part of the Shiny for Python code :
def server(input: Inputs, output: Outputs, session: Session):

#Loading the treat CSV file
    @output
    @render.table
    def Treattable():
        infile = Path(__file__).parent / "data/treat_data.csv"
        treat = pd.read_csv(infile)
        return treat
    
#Trying to display the Linear Regression Graphs and Waterfall chart on the Dashboard by creating
#PNG images and using them to display them on the dashboard
    @output
    @render.plot
    def Data_test():
        x, y = data_separation(treat)
        x = pd.DataFrame(x)
        x = x.fillna(0)
        x = x.dropna()
        X_train, X_test, Y_train, Y_test = data_split(x,y)
        lr, y_lr_train_pred, y_lr_test_pred = LR(X_train, X_test, Y_train)
        linear_regression_plot = plot_linear_regression_results(Y_train, y_lr_train_pred)
        return linear_regression_plot
    
    

#Function for WaterFall chart for Linear Regression
    @output
    @render.plot
    def WaterfallPNG():
        x, y = data_separation(treat)

        x = pd.DataFrame(x)
        x = x.fillna(0)
        x = x.dropna()
        X_train, X_test, Y_train, Y_test = data_split(x,y)
 
        lr, y_lr_train_pred, y_lr_test_pred = LR(X_train, X_test, Y_train)
        Water = create_shap_waterfall_chart(lr, x, X_test, sample_index=14, max_display=14)
        
        return Water

    @output
    @render.plot
    def Random_Forest_plot():
        x, y = data_separation(treat)
        x = pd.DataFrame(x)
        x = x.fillna(0)
        x = x.dropna()
        X_train, X_test, Y_train, Y_test = data_split(x,y)
        rf, y_rf_train_pred, y_rf_test_pred = RF(X_train,Y_train, X_test, max_depth=2, random_state=100)
        RF_plot = plot_RF_results(Y_train, y_rf_train_pred)
        return RF_plot
    
    @output
    @render.plot
    def WaterRF():
        x, y = data_separation(treat)
        x = pd.DataFrame(x)
        x = x.fillna(0)
        x = x.dropna()
        X_train, X_test, Y_train, Y_test = data_split(x,y)
        rf, y_rf_train_pred, y_rf_test_pred = RF(X_train,Y_train, X_test, max_depth=2, random_state=100)
        RFWater = create_shap_waterfall_chart(rf, x, X_test, sample_index=14, max_display=14)
        return RFWater


#Function for nav"Other Types of SHAP charts" to display two lists of the
#positive and negative features
    @output
    @render.text
    def positive_negative():
         x, y = data_separation(treat)
         x = pd.DataFrame(x)
         x = x.fillna(0)
         x = x.dropna()
         X_train, X_test, Y_train, Y_test = data_split(x,y)
        # Drop rows containing NaN values
 
         lr, y_lr_train_pred, y_lr_test_pred = LR(X_train, X_test, Y_train)
         positive_feature_names, negative_feature_names = Split_and_Shap(lr, x, X_test, sample_index=14, max_display=14)
        # Return the two lists as a tuple
         return f" The positive features are : {positive_feature_names}\n & the negative features are :{negative_feature_names}"
    
#Function for nav"Other Types of SHAP charts" to display two other SHAP chart "Beeswarm" and "Violin"
    @output
    @render.plot
    def plot_bee():
        x, y = data_separation(treat)
        x = pd.DataFrame(x)
        x = x.fillna(0)
        x = x.dropna()
        X_train, X_test, Y_train, Y_test = data_split(x, y)
        lr, y_lr_train_pred, y_lr_test_pred = LR(X_train, X_test, Y_train)
        bee = shap_beeswarm_plot(lr, x, X_test, sample_index=14, max_display=14)
        return bee  # Return the SHAP graph

#Same but for Violin SHAP chart
    @output
    @render.plot
    def plot_violin():
        x, y = data_separation(treat)
        x = pd.DataFrame(x)
        x = x.fillna(0)
        x = x.dropna()
        X_train, X_test, Y_train, Y_test = data_split(x, y)
        lr, y_lr_train_pred, y_lr_test_pred = LR(X_train, X_test, Y_train)
        violin = shap_violin_plot(lr, x, X_test, sample_index=14, max_display=14)

        return violin

#Shiny for Python function to display Patient info on the Joblib prediction tab
    """@output
    @render.text
    @reactive.event(input.send3, ignore_none=False)
    def patient_Row():
        patient_id = input.patient_row
        pred = Joblib(patient_id())
        return f"Prediction for the Patient {pred}"""
    

#Tab Joblib Prediction, Text to display Prediction made with Model and dataset
    @output
    @render.text
    @reactive.event(input.send3, ignore_none=False)
    def Pred():
        selected_row = input.patient_row
        prediction = Joblib(selected_row())
        return prediction
    
    @output
    @render.plot
    @reactive.event(input.send3, ignore_none=False)
    def Pred_plot():
        selected_row = input.patient_row
        plot = pred_plot(selected_row())
        return plot



#Shiny for Python for the What if navigation bar and what is inside
    @output
    @render.plot
    @reactive.event(input.pred, ignore_none=False)
    def new_LR_plt():
        age = input.age.get()
        blood = input.blood_pressure.get()
        gender = input.gender
        diabetes = input.diabetes
        model = LinearRegression()
        #Setting up x and y :
        np.random.seed(19680801)
        x = age * np.random.randn(437)
        y = blood * np.random.randn(437)
        X_train, X_test, Y_train, Y_test = data_split(x, y)
        lr, y_lr_train_pred, y_lr_test_pred = LR(X_train, X_test, Y_train)
        plot = plot_linear_regression_results(Y_train, y_lr_train_pred)
        return plot



#Shiny for Python function for the What if tab
    @output
    @render.plot
    def Current():
         #Setting up x and y :
        x, y = data_separation(treat)
        x = pd.DataFrame(x)
        x = x.fillna(0)
        x= x.dropna()
        X_train, X_test, Y_train, Y_test = data_split(x, y)
        lr, y_lr_train_pred, y_lr_test_pred = LR(X_train, X_test, Y_train)
        Cur = plot_linear_regression_results(Y_train, y_lr_train_pred)
        return Cur

#Shiny for Python function to display Patient info   
    @output
    @render.table
    @reactive.event(input.send, ignore_none=False)
    def patient_table():
        patient_ids = input.patient_id
        response = requests.get('{}/{}/{}'.format(BASE_URL, 'Patient', patient_ids()))

        patient_df = pd.json_normalize(response.json())[['id', 'gender', 'birthDate']]
        patient_df = patient_df.astype({'birthDate': 'datetime64[ns]'})
        return patient_df
    

#Shiny for Python function to display Patient history   
    @output
    @render.table
    @reactive.event(input.send2, ignore_none=False)
    def history() :
        patient_id=input.patient_id
        patient2 =input.patient2
        code = input.snowmed
        response = requests.get('{}/{}?patient={}&code={}'.format(BASE_URL, 'Observation', patient_id(), patient2(), code()))
        history_df = pd.json_normalize(response.json())
        return history_df
    



app = App(app_ui, server)