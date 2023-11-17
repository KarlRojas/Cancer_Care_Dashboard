import numpy as np
import json
import requests_fhir as requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import shap
import seaborn as sns
import joblib
from sklearn.feature_selection import SelectKBest, f_classif
from requests.exceptions import RequestException
from sklearn import *
from shinywidgets import output_widget, render_widget
from asyncio import sleep
import plotly.graph_objs as go
import shinyswatch
from htmltools import css
from flask import send_file
import plotly.express as px
import streamlit as st
from pathlib import Path
from typing import List 
from shiny.types import NavSetArg
from sklearn.preprocessing import StandardScaler
BASE_URL = 'https://test/fhir'
