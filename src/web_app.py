import os
import numpy as np
import joblib
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

#from sklearn.ensemble import RandomForestClassifier

# Load the saved model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'notebooks', 'best_model.pkl')

model_data = joblib.load(model_path)
model = model_data["model"]
scaler = model_data["scaler"]
feature_names = model_data["feature_names"]

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Layout
app.layout = html.Div([
    html.H1("Student Grade Classification Predictor"),

    html.Br(), html.Br(),

    html.Label("Age:"),
    dcc.Input(id='age', type='number', value=18, step=1),

    html.Br(), html.Br(),

    html.Label("Gender:"),
    dcc.Dropdown(
        id='gender',
        options=[{'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}],
        value='Male'
    ),

    html.Br(), html.Br(),

    html.Label("Parental Education Level:"),
    dcc.Dropdown(
        id='parental_education',
        options=[{'label': "None", 'value': 0},
                 {'label': "High School", 'value': 1},
                 {'label': "Bachelor's", 'value': 2},
                 {'label': "Master's", 'value': 3},
                 {'label': "PhD", 'value': 4}],
        value=0
    ),

    html.Br(), html.Br(),

    html.Label("Weekly Study Time (hrs):"),
    dcc.Input(id='study_time', type='number', value=0.0, step=0.01),

    html.Br(), html.Br(),

    html.Label("Absences:"),
    dcc.Input(id='absences', type='number', value=0, step=1),

    html.Br(), html.Br(),

    html.Label("Tutoring:"),
    dcc.Dropdown(id='tutoring', options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], value=0),

    html.Br(), html.Br(),

    html.Label("Parental Support:"),
    dcc.Dropdown(id='parental_support', options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], value=0),

    html.Br(), html.Br(),

    html.Label("Extracurricular Activities:"),
    dcc.Dropdown(id='extracurricular', options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], value=0),

    html.Br(), html.Br(),

    html.Label("Sports:"),
    dcc.Dropdown(id='sports', options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], value=0),

    html.Br(), html.Br(),

    html.Label("Music:"),
    dcc.Dropdown(id='music', options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], value=0),

    html.Br(), html.Br(),

    html.Label("Volunteering:"),
    dcc.Dropdown(id='volunteering', options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], value=0),

    html.Br(), html.Br(),

    html.Button('Predict Grade', id='predict-btn'),
    html.Div(id='output')
])

# Callback
@app.callback(
    Output('output', 'children'),
    Input('predict-btn', 'n_clicks'),
    Input('age', 'value'),
    Input('gender', 'value'),
    Input('parental_education', 'value'),
    Input('study_time', 'value'),
    Input('absences', 'value'),
    Input('tutoring', 'value'),
    Input('parental_support', 'value'),
    Input('extracurricular', 'value'),
    Input('sports', 'value'),
    Input('music', 'value'),
    Input('volunteering', 'value')
)
def predict_grade(n_clicks, age, gender, parental_education, study_time, absences, tutoring, parental_support, extracurricular, sports, music, volunteering):
    if not n_clicks:
        return ""

    if None in [age, gender, parental_education, study_time, absences, tutoring, parental_support, extracurricular, sports, music, volunteering]:
        return "Please fill in all fields."

    try:
        # Preprocessing
        gender_mapping = {"Male": 0, "Female": 1}
        gender_num = gender_mapping.get(gender)

        support_score = (parental_support / 4) * 0.45 + (parental_education / 4) * 0.1 + tutoring * 0.45
        at_risk = int((study_time < 5) and (absences > 10) and (support_score < 0.5))
        activity_score = extracurricular + sports + music + volunteering
        absence_study_interaction = absences * study_time
        parental_support_study_time = parental_support * study_time
        study_time_per_absence = study_time / (absences + 1)
        support_per_activity = support_score / (activity_score + 1)
        absences_squared = absences ** 2
        study_time_squared = study_time ** 2
        tutoring_study_interaction = tutoring * study_time
        multiple_activities = int(activity_score > 1)

        input_vector = [
            age, gender_num, parental_education, study_time, absences, tutoring, parental_support,
            extracurricular, sports, music, volunteering, support_score, at_risk, activity_score,
            absence_study_interaction, parental_support_study_time, study_time_per_absence,
            support_per_activity, absences_squared, study_time_squared,
            tutoring_study_interaction, multiple_activities
        ]

        # Scale and predict
        input_array = np.array([input_vector])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]

        grade_mapping = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
            4: "F"
        }

        predicted_grade = grade_mapping.get(int(prediction), "Unknown")
        return f"Predicted Grade Classification: {predicted_grade}"

    except Exception as e:
        return f"Error during prediction: {str(e)}"


#Run the app locally
'''if __name__ == "__main__":
    app.run(debug=True)
'''