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

#new navigation bar 
def nav_controls(prefix: str) -> List[NavSetArg]:
    return [
        ui.nav_spacer(),
        ui.nav("Patient Information", prefix + ": Patient Informations",
               x.ui.card(
                   x.ui.card_header("Patient Information"),
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
        ui.nav("Linear Regression & Random Forest", prefix + ": Linear Regression & Random Forest",
               ui.row(
                    ui.column(
                        6,
                        x.ui.card(
                            x.ui.card_header("Linear Regression "),
                            ui.output_plot("Linear_Regression"),
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
                "BeeSwarm & Violin Graphs",prefix + ": BeeSwarm & Violin Graphs",
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
                "Joblib Prediction", prefix + ": Joblib Prediction",
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
                "Treatment Plans",  prefix + ": Treatment Plans",
                ui.div(
                    ui.input_select(
                        "x", label="Variable",
                        choices=["total_bill", "tip", "size"]
                    ),
                    ui.input_select(
                        "color", label="Color",
                        choices=["smoker", "sex", "day", "time"]
                    ),
                    class_="d-flex gap-3",

                ),
                output_widget("my_widget"),
                
            ),
            ui.nav(
                "Feedback and Support", prefix + ": Feedback and Support",
            )
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
        response = patient_data(patient_ids())
        return response
    

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