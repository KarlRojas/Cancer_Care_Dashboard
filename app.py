from shiny import *
from Functions import *
import shiny as x
from Librairies import *
#import seaborn as sns


#Opening the Treatment CSV file
infile = Path(__file__).parent / "data/treat_data.csv"
treat = pd.read_csv(infile)

#Loading the pre-trained model and Opening the Simulated Data
model = joblib.load('model.joblib')
data = pd.read_csv('simulated_data.csv')

#Opening the Snomed CSV file
infiles = Path(__file__).parent / "data/snomed.csv"
snomed = pd.read_csv(infiles)
labels = snomed['label'].unique()

#new navigation bar 
def nav_controls(prefix: str) -> List[NavSetArg]:
    return [
        ui.nav_spacer(),
        ui.nav("Patient History", prefix + " : Patient Informations",
               ui.row(
                   ui.column(
                       6,
                       x.ui.card(
                           x.ui.card_header("Patient Information"),
                           ui.input_numeric("patient_id", "Enter the Patient ID", 2, min=1, max=1000000000),
                           ui.p(ui.input_action_button("send", "Enter", class_="btn-primary")),
                           ui.output_table("patient_table"),
                           fill = True,
                           height = "300px",
               ),
                   ),

                   ui.column(
                       6,
                       x.ui.card(
                           x.ui.card_header("Patient History"),
                           ui.input_selectize(
                           "selected_label",
                           "Choose a label", 
                           {label: label for label in labels},
                           multiple=False
                    ),
                    fill = True,
                    height = "300px",
               ),
                   ),
               x.ui.card(
                   x.ui.card_header("Patient plot"),
                   ui.output_plot("history"),
                   fill = True,
                   full_screen =True,
               ),
                             
               ),
               
        ),
        ui.nav("Home Page", prefix + " : Home Page",
                   ui.h2("Welcome to the Home Page of the Prediction Dashboard"),
                   ui.h5("Currently logged in as : ... Hospital : ... Number of assigned patients : ..."),
                   ui.p("To get started please select a patient : "),
                   ui.input_numeric("patient_ID", "Enter the Patient ID", 0, min=1, max=1000000000),
                   ui.p(ui.input_action_button("sends", "Enter", class_="btn-primary")),
                   #ui.output_table("Patient_data"),
                   ui.a("Choosing a patient from the list", href="https://shiny.posit.co/py/"),
                   ),
        
            ui.nav("Overview", prefix + " : Patient Overview",
                   ui.row(
                       ui.column(4,
                                 x.ui.card(
                                     #ui.input_numeric("patient_ID", "Enter the Patient ID", 0, min=1, max=1000000000),
                                     #ui.p(ui.input_action_button("sends", "Enter", class_="btn-primary")),
                                     x.ui.card_header("Patient Information"),
                                     ui.output_table("Patient_data"),
                                 ),
                                 ),
                       ui.column(4,
                                 x.ui.card(
                                     x.ui.card_header("Diagnosis"),
                                 ),
                                 ),
                       ui.column(4,
                                 x.ui.card(
                                     x.ui.card_header("Calendar"),
                                     ui.output_text("Date"),

                                 ),
                                 ),
                   ui.row(
                       ui.column(
                           6,
                           x.ui.card(
                               x.ui.card_header("Linear Regression Prediction"),
                               ui.output_plot("Linear_Regression"),
                               ui.output_text("positive_negative2"),
                               ui.output_text("Pred"),
                               full_screen =True,
                           ),
                           
                       ),

                       ui.column(
                           6,
                           x.ui.card(
                               x.ui.card_header("Random Forest Prediction"),
                               ui.output_plot("Random_Forest_plot"),
                               ui.output_text("positive_negative"),
                               ui.output_text("Pred2"),
                               full_screen =True,
                           ),
                       ),

                    ),
                    
                       
                   ),

            ),
                   
        
        ui.nav(
            "What if",prefix + ":What if",
                ui.row(
                    ui.column(8,
                        x.ui.card(
                            x.ui.card_header("What if"),

                            #ui.p(ui.input_action_button("pred", "Create a new Prediction", class_="btn-primary")),
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
                                animate= True
                            ),
                            ui.input_slider(
                                "blood_pressure",
                                "Blood_Pressure",
                                60,
                                150,
                                100,
                                step=0.01,
                                animate=True
                            ),
                            ui.input_slider(
                                "blood_pressure",
                                "Blood_Pressure",
                                60,
                                150,
                                100,
                                step=0.01,
                                animate=True
                            ),
                            ui.input_slider(
                                "blood_pressure",
                                "Blood_Pressure",
                                60,
                                150,
                                100,
                                step=0.01,
                                animate=True
                            ),
                        ),
                    ),
                    ui.column(4,
                        x.ui.card(
                            x.ui.card_header("Other Metrics"),
                            ui.input_radio_buttons(
                                "othermetrics",
                                "Select metric :",
                                {
                                    "M1": "Blood",
                                    "M2": "Blood",
                                 },

                            ),

                        ),
                    ),
                ),
            ui.row(
                x.ui.card(
                    x.ui.card_header("Prediction"),
                    ui.row(
                        ui.column(10,
                            ui.input_radio_buttons(
                                "model",
                                "Select model :",
                                    {
                                    "R1":"Logistic Regression Model",
                                    "R2": "Random Forest Model",
                                 },
                                inline=True,
                            ),
                        ),
                    ),
                    ui.row(
                        ui.column(6,
                                  ui.output_plot("Prediction"),
                                  full_screen=True,
                                  ),
                        ui.column(6,
                                  ui.output_text("Feedback"),
                                  full_screen=True,
                                  ),
                    ),
                ),
            ),

            ui.row(
                x.ui.card(
                    ui.card_header("Prediction Average"),
                    ui.output_plot("PlotAverage"),
                    full_screen=True,
                ),
            ),
        ),
            
         ui.nav("Joblib Prediction", prefix + ": Joblib Prediction",
                   ui.row(
                       ui.column(6,
                                 x.ui.card(
                                     x.ui.card_header("Patient row"),
                                     ui.input_numeric("patient_row", "Enter the Patient row", 1, min=1, max=len(data)),
                                     ui.p(ui.input_action_button("send3", "Enter", class_="btn-primary")),
                                     ui.output_text("patient_Row"),
                                     height ="200px",
                                 ),
                       ),
                        ui.column(6,
                                  x.ui.card(
                                      x.ui.card_header("Predictions results"),
                                      #ui.output_text("Pred"),
                                  ),
                        ),

                        ui.column(6,
                                  x.ui.card(
                                      x.ui.card_header("Predictions Plot"),
                                      ui.output_plot("Pred_plot"),
                                      full_screen =True,
                        ),
                                  ),
                        ui.column(6,
                                  x.ui.card(
                                      x.ui.card_header("Waterfall Chart"),
                                      ui.output_plot("Waterfallpred"),
                                      full_screen =True,
                                     fill = True, 
                        ),
                                  ),
                    ),
                        
                        
            ),
           
        ui.nav(
                "Treatment Plans",  prefix + ": Treatment Plans",
            ),

            
        
        

            


    ]


app_ui = ui.page_navbar(
    *nav_controls("Page"),
    shinyswatch.theme.united(),
    title="Predicto",
    id="navbar_id",


)

#Server part of the Shiny for Python code :
def server(input: Inputs, output: Outputs, session: Session):

    @reactive.Effect
    def _():
        print("Current navbar page: ", input.navbar_id())

#Shiny for Python function to display the Linear Regression 
    @output
    @render.plot
    def Linear_Regression():
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

#Function for Random Forest chart
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

#Function to display the WaterFall chart of the Random Forest 
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
    


    
#Function for nav"Other Types of SHAP charts" to display two lists of the
#positive and negative features
    @output
    @render.text
    def positive_negative2():
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


#Tab Joblib Prediction, Text to display Prediction made with Model and dataset
    @output
    @render.text
    @reactive.event(input.send3, ignore_none=False)
    def Pred():
        selected_row = input.patient_row
        prediction = Joblib(selected_row())
        return prediction
    
#Tab Joblib Prediction, Text to display Prediction made with Model and dataset
    @output
    @render.text
    @reactive.event(input.send3, ignore_none=False)
    def Pred2():
        selected_row = input.patient_row
        prediction = Joblib(selected_row())
        return prediction

#Function for the Tab Prediction, displays a histogram plot
    @output
    @render.plot
    @reactive.event(input.send3, ignore_none=False)
    def Pred_plot():
        selected_row = input.patient_row
        plot = pred_plot(selected_row())
        return plot

    @output
    @render.plot
    def Waterfallpred():
        x, y = separation_pred(data)
        x = pd.DataFrame(x)
        x = x.fillna(0)
        x = x.dropna()
        X_train, X_test, Y_train, Y_test = data_split(x,y)
        lr, y_lr_train_pred, y_lr_test_pred = LR(X_train, X_test, Y_train)
        Water = create_shap_waterfall_chart(lr, x, X_test, sample_index=14, max_display=5)
        return Water
    

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
    @reactive.event(input.sends, ignore_none=False)
    def Patient_data():
        patient_ids = input.patient_ID
        response = patient_data(patient_ids())
        return response
    

#Shiny for Python function to display Patient history


    @output
    @render.plot
    @reactive.event(input.send, ignore_none=False)
    def history():
        patient_id=input.patient_id()
        code = input.selected_label()

        # Sélection d'une Entrée SNOMED
        snomed_entry = snomed.loc[snomed['label'] == code].iloc[0]

        # Requête pour Récupérer les Mesures de Patients
        response = requests.get('{}/{}?patient={}&code={}'.format(BASE_URL, 'Observation', patient_id, snomed_entry['code']))

        # Normalisation des Données de Mesures
        history_df = pd.json_normalize(response.json(), record_path='entry')[
            ['resource.subject.reference', 'resource.effectiveDateTime', 'resource.valueQuantity.value']]
        history_df['resource.valueQuantity.value'] = pd.to_numeric(history_df['resource.valueQuantity.value'], errors='coerce')
        history_df = history_df.astype(
            {'resource.effectiveDateTime': 'datetime64[ns]', 'resource.valueQuantity.value': 'float64'})

        # Tracé des Mesures pour Chaque Patient
        for patient in history_df['resource.subject.reference'].value_counts().index:
            patient_history_df = history_df.loc[history_df['resource.subject.reference'] == patient]
            sns.lineplot(x=patient_history_df['resource.effectiveDateTime'],
                     y=patient_history_df['resource.valueQuantity.value'], label=patient)

        # Personnalisation du Graphique et Affichage

    @output
    @render.text
    def date():
        # Get the current date and time
        current_datetime = datetime.now()

        # Format the current date as a string with a different format
        current_date = current_datetime.strftime("%A, %B %d, %Y")

        return current_date


app = App(app_ui, server)