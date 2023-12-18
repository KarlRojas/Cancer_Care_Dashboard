from Librairies import *

#Function to load Patient Data
def patient_data(patient_id):
    try:
        response = requests.get('{}/{}/{}'.format(BASE_URL, 'Patient', patient_id))
        if response.status_code == 200:
            # Normalize and select relevant columns
            patient_df = pd.json_normalize(response.json())[['id', 'gender', 'birthDate']]
            # Convert birthDate column to datetime
            patient_df = patient_df.astype({'birthDate': 'datetime64[ns]'})
            return patient_df
        else:
            print("Failed to fetch patient data. Status code:", response.status_code)
            return None
    except RequestException as e:
        print("Error:", e)
        return None





#Function to Load Snomed CSV File
def read_and_process_snomed_csv(file_path):
    # Read the CSV file
    snomed = pd.read_csv(file_path)
    # Convert 'code' column to string type
    snomed = snomed.astype({'code': 'string'})
    # Display info and the first 5 rows
    return snomed

#Function to Load Treatment CSV File
def treatment_file(file_path):
    # Read the CSV file
    treat = pd.read_csv(file_path)
    # Convert 'treat_start_date' column to datetime
    treat['treat_start_date'] = pd.to_datetime(treat['treat_start_date'])
    # Convert 'record_id' column to string type
    treat['record_id'] = treat['record_id'].astype(str)
    # Display info and the DataFrame
    print(treat.info())
    print(treat)
    return treat

#Function for the X and Y variables
def data_separation(treat_df):
    y = treat_df['cvscore']
    x = treat_df.drop(['cvscore','treat_start_date'], axis = 1)
    return x, y

#Function for the X and Y variables
def separation_pred(data):
    y = data['days_from_diag']
    x = data.drop(['days_from_diag'], axis = 1)
    return x, y

#Function for Data splitting 
def data_split(x, y, test_size=0.2, random_state=100):
    X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=test_size, random_state=random_state)
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.fit_transform(X_test)
    return X_train, X_test, Y_train, Y_test

#Function for Data splitting 2
def data_split2(x, y, test_size=0.2, random_state=100):
    X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=test_size, random_state=random_state)
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.fit_transform(X_test)
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test


#Function for Linear Regression
def LR(X_train, X_test, Y_train):
    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    y_lr_train_pred = lr.predict(X_train)
    y_lr_test_pred = lr.predict(X_test)
    return lr, y_lr_train_pred, y_lr_test_pred

#Function to evaluate the Linear Regression
def evaluate_linear_regression(Y_train, y_lr_train_pred, Y_test, y_lr_test_pred):
    lr_train_mse = mean_squared_error(Y_train, y_lr_train_pred)
    lr_train_r2 = r2_score(Y_train, y_lr_train_pred)
    lr_test_mse = mean_squared_error(Y_test, y_lr_test_pred)
    lr_test_r2 = r2_score(Y_test, y_lr_test_pred)
    print('LR MSE (training): ', lr_train_mse)
    print('LR R2 (training): ', lr_train_r2)
    print('LR MSE (test): ', lr_test_mse)
    print('LR R2 (test): ', lr_test_r2)
    lr_result = pd.DataFrame(['Linear regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
    return lr_result

#Function for Random Forest
def RF(X_train,Y_train, X_test, max_depth=2, random_state=100 ):
    rf = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
    rf.fit(X_train, Y_train)
    y_rf_train_pred = rf.predict(X_train)
    y_rf_test_pred = rf.predict(X_test)
    return rf, y_rf_train_pred, y_rf_test_pred

#Function to evaluate the Random Forest 
def train_and_evaluate_random_forest(Y_train, y_rf_train_pred, Y_test, y_rf_test_pred):
    rf_train_mse = mean_squared_error(Y_train, y_rf_train_pred)
    rf_train_r2 = r2_score(Y_train, y_rf_train_pred)
    rf_test_mse = mean_squared_error(Y_test, y_rf_test_pred)
    rf_test_r2 = r2_score(Y_test, y_rf_test_pred)
    print('RF MSE (training): ', rf_train_mse)
    print('RF R2 (training): ', rf_train_r2)
    print('RF MSE (test): ', rf_test_mse)
    print('RF R2 (test): ', rf_test_r2)
    return rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2

#Function to plot the Linear Regression
def plot_linear_regression_results(Y_train, y_lr_train_pred):
    z = np.polyfit(Y_train, y_lr_train_pred, 1)
    p = np.poly1d(z)
    scatter = sns.scatterplot(x=Y_train, y=y_lr_train_pred, alpha=0.3)
    plt.plot(Y_train, p(Y_train), color='red', linewidth=2)
    plt.xlabel('Actual CVS Score')
    plt.ylabel('Predicted CVS Score')
    plt.title('Linear Regression: Actual vs. Predicted CVS Score')

#Function to plot the Linear Regression
def plot_RF_results(Y_train, y_rf_train_pred):
    plt.figure(figsize=(5, 5))
    plt.scatter(x=Y_train, y=y_rf_train_pred, alpha=0.3)
    z = np.polyfit(Y_train, y_rf_train_pred, 1)
    p = np.poly1d(z)
    plt.plot(Y_train, p(Y_train), color='red')
    plt.xlabel('Actual CVS Score')
    plt.ylabel('Predicted CVS Score')
    plt.title('Random Forest: Actual vs. Predicted CVS Score')

#Function to plot a SHAP Waterfall chart
def create_shap_waterfall_chart(model, x, X_test, sample_index=14, max_display=14):
    # Get feature names from the DataFrame
    feature_names = x.columns.tolist()
    # Create a SHAP explainer
    explainer = shap.Explainer(model, X_test, feature_names=feature_names)
    # Calculate SHAP values
    shap_values = explainer(X_test)
    # Create a SHAP waterfall chart for a specific instance
    plt.tight_layout()
    shap.plots.waterfall(shap_values[sample_index], max_display=max_display, show=False)


#Function to plot a BeeSwarm SHAP chart
def shap_beeswarm_plot(model, x, X_test, sample_index, max_display=20):
    # Create a SHAP beeswarm plot
    feature_names = x.columns.tolist()
    explainer = shap.Explainer(model, X_test, feature_names=feature_names)
    shap_values = explainer(X_test)
    plt.title("Impact of the features")
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)

 


#Function to plot a Violin SHAP chart
def shap_violin_plot(model, x, X_test, sample_index, max_display=20):
    # Create a SHAP violin plot
    feature_names = x.columns.tolist()
    explainer = shap.Explainer(model, X_test, feature_names=feature_names)
    shap_values = explainer(X_test)
    shap.plots.violin(shap_values, show=False)



#Function for the Shap chart nav to create both the Beeswarm and Violin chart
#and to  retrieve the positive and negative features in a list
def Split_and_Shap(model, x, X_test, sample_index, max_display=14):
    # Get feature names from the DataFrame
    feature_names = x.columns.tolist()
    
    # Create a SHAP explainer
    explainer = shap.Explainer(model, X_test, feature_names=feature_names)
    
    # Calculate SHAP values
    shap_values = explainer(X_test)
    
    # Create two SHAP charts so that we can retrieve the features
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    shap.plots.violin(shap_values, show=False)

    # Initialize empty lists for positive and negative feature names
    positive_feature_names = []
    negative_feature_names = []

    # Iterate through the SHAP values and split them
    for i, value in enumerate(shap_values):
        if i < len(feature_names):  # Ensure that the index is within the bounds of feature_names
            feature_name = feature_names[i]  # Get the feature name from the corresponding index

            if any(value.values > 0):  # Check if any value in the Explanation is positive
                positive_feature_names.append(feature_name)
            elif any(value.values < 0):  # Check if any value in the Explanation is negative
                negative_feature_names.append(feature_name)

    # Return positive and negative feature names as two separate lists
    return positive_feature_names, negative_feature_names


#Function for the Joblib Method and to realize predictions for each patients
def Joblib(selected_row):
 # Load the pre-trained model
    model = joblib.load('model.joblib')

    # Read the CSV file into a DataFrame
    data = pd.read_csv('simulated_data.csv')

    # Define the number of features to select (304 in your case)
    num_features_to_select = 304

    # Extract the selected patient's data
    patient_data = data.iloc[selected_row - 1]  # Assuming selected_row is the row number you want to predict

    # Extract the 'days_from_diag' column from the patient's data
    days_from_diag = patient_data['side_effect_constip']

    # Find the index (position) of the 'days_from_diag' column
    days_from_diag_idx = data.columns.get_loc('side_effect_constip')

    # Extract features starting from 'days_from_diag' and select the next 304 columns
    selected_features = patient_data[days_from_diag_idx + 1:days_from_diag_idx + num_features_to_select + 1]

    # Create a new DataFrame with the selected features
    selected_data = pd.DataFrame(selected_features).T

    # Make predictions
    prediction = model.predict(selected_data)
    per = round(prediction[0] * 100, 2)

    return f"The risk of dying within 30 days for patient {selected_row} is {per:.2f}% according to my model."



def pred_plot(selected_row):
    # Load the pre-trained model
    model = joblib.load('model.joblib')
    # Read the CSV file into a DataFrame
    data = pd.read_csv('simulated_data.csv')
    num_features_to_select = 304

    # Extract the selected patient's data
    patient_data = data.iloc[selected_row - 1]
    # Extract the 'days_from_diag' column from the patient's data
    days_from_diag = patient_data['side_effect_constip']

    # Find the index (position) of the 'days_from_diag' column
    days_from_diag_idx = data.columns.get_loc('side_effect_constip')

    # Extract features starting from 'days_from_diag' and select the next 304 columns
    selected_features = patient_data[days_from_diag_idx + 1:days_from_diag_idx + num_features_to_select + 1]

    # Create a bar plot to visualize the 'days_from_diag' values
    plt.figure(figsize=(8, 6))
    plt.hist(selected_features, bins=20)
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.title(f'Days from Diagnosis for Patient {selected_row}')
