from shiny import *
from Functions import *
import shiny as x
from Librairies import *


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
        


        ui.nav("Patient Information", prefix + ": Patient Informations",
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
        ui.nav("Linear Regression & Random Forest", prefix + ": Linear Regression & Random Forest",
               ui.row(
                    ui.column(
                        6,
                        x.ui.card(
                            x.ui.card_header("Linear Regression "),
                            ui.output_plot("Linear_Regression"),
                            full_screen =True,
                        ),
                        x.ui.card(
                            x.ui.card_header("Linear Regression Waterfall chart"),
                            ui.output_plot("WaterfallPNG"),
                            full_screen =True,
                        ),
                    ),
                    ui.column(
                        6,
                        x.ui.card(
                            x.ui.card_header("Random Forest"),
                            ui.output_plot("Random_Forest_plot"),
                            full_screen =True,
                        ),

                        x.ui.card(
                            x.ui.card_header("Waterfall Random Forest"),
                            ui.output_plot("WaterRF"),
                            full_screen =True,
                        ),
                    ),
                ),
               ),
        ui.nav(
                "BeeSwarm & Violin Graphs",prefix + ": BeeSwarm & Violin Graphs",
                x.ui.card(
                    x.ui.card_header("Positive and negative SHAP features"),
                    ui.output_text("positive_negative"),
                    fill=True,
                    height = "300px",
                ),
                ui.row(
                    ui.column(
                        6,
                        x.ui.card(
                            x.ui.card_header("Beeswarm Chart :"),
                            ui.output_plot("plot_bee"),
                            fill=True, 
                            full_screen=True, 
                        ),
                    ),
                    ui.column(
                        6,
                        x.ui.card(
                            x.ui.card_header("Violin Chart :"),
                            ui.output_plot("plot_violin"),
                            fill=True, 
                            full_screen=True, 
                        ),
                    ),
                ),
            ),

        ui.nav(
                "What if analysis",prefix + ": What if analysis",
                    x.ui.card(
                    x.ui.card_header("What if"),
                    ui.p(ui.input_action_button("pred", "Create a new Prediction!", class_="btn-primary")),
                    ui.input_slider(
                        "age",
                        "Age",
                        0,
                        120,
                        65,
                        post =" y.o",
                    ),
                    ui.input_slider(
                        "blood_pressure",
                        "Blood_Pressure",
                        60,
                        150,
                        100,
                        step=0.01,
                        animate = True,
                        post = " mmHg",
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
                                      ui.output_text("Pred"),
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
            ui.nav(
                "Feedback and Support", prefix + ": Feedback and Support",
            ),
            ui.nav(
                "Test for main page", prefix + ": Main Page",
                x.ui.card(
                    ui.row(
                        ui.column(
                            6,
                            x.ui.card(
                                x.ui.card_header("Choosing a patient"),
                                ui.input_numeric("id", "Enter the Patient ID", 2, min=1, max=1000000000),
                                ui.p(ui.input_action_button("sending", "Enter", class_="btn-primary")),
                            ),
                        ),
                        ui.column(
                            6,
                            x.ui.card(
                                x.ui.card_header("Patient Info"),
                                ui.output_table("patient_tab"),
                            ),
                        ),
                    ),
                    ui.row(
                        ui.column(
                            6,
                            x.ui.card(
                                x.ui.card_header("Linear Regression"),
                                ui.output_plot("Linear_Regression"),
                                full_screen =True,

                               


                            ),
                        ),
                        ui.column(
                            6,
                            x.ui.card(
                                x.ui.card_header("Random Forest"),

                                x.ui.card_header("Prediction"),

                                x.ui.card_header("Positive and Negative SHAP features"),

                            ),
                        ),
                    ),
                ),

            ),
    ]


app_ui = ui.page_navbar(
    *nav_controls("Page"),
    shinyswatch.theme.darkly(),
    title="AI Dashboard for Cancer Care",
    id="navbar_id",
)

#Server part of the Shiny for Python code :
def server(input: Inputs, output: Outputs, session: Session):

    @reactive.Effect
    def _():
        print("Current navbar page: ", input.navbar_id())

    @output
    @render.text
    def current_date():
        current_dates = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Current Date and Time: {current_date}"
    
    @output
    @render.text
    def Hoem():
        return "Welcome to Dashboard name, username"

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
    @reactive.event(input.send2, ignore_none=False)
    def patient_table():
        patient_ids = input.patient_id
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


#Shiny for Python function to display Patient info   
    @output
    @render.table
    @reactive.event(input.button, ignore_none=False)
    def patient_tab():
        patient_is = input.id
        responses = patient_data(patient_is())
        return responses
    


app = App(app_ui, server)