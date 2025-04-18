import os
import joblib
import numpy as np
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
from dash import callback_context

# Load the saved model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'artifacts', 'best_model.pkl')

model_data = joblib.load(model_path)
model = model_data["model"]
scaler = model_data["scaler"]
feature_names = model_data["feature_names"]

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Student Grade Class Predictor"
server = app.server

# Layout
app.layout = html.Div([
    html.H1("Student Grade Class Predictor"),

    html.Div(className='container', children=[
        html.Label("Age:"),
        dcc.Input(id='age', type='number', value=18, step=1, min=0),

        html.Label("Gender:"),
        dcc.Dropdown(
            id='gender',
            options=[{'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}],
            value='Male'
        ),

        html.Label("Parental Education Level:"),
        dcc.Dropdown(
            id='parental_education',
            options=[{'label': "None", 'value': 0},
                     {'label': "High School", 'value': 1},
                     {'label': "College", 'value': 2},
                     {'label': "Bachelor's", 'value': 3},
                     {'label': "Higher Study", 'value': 4}],
            value=0
        ),

        html.Label("Weekly Study Time (hours):"),
        dcc.Input(id='study_time', type='number', value=0, step=1, min=0),

        html.Label("Absences:"),
        dcc.Input(id='absences', type='number', value=0, step=1, min=0),

        html.Label("Tutoring:"),
        dcc.Dropdown(id='tutoring', options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], value=0),

        html.Label("Parental Support:"),
        dcc.Dropdown(
            id='parental_support',
            options=[
                {'label': 'None', 'value': 0},
                {'label': 'Low', 'value': 1},
                {'label': 'Moderate', 'value': 2},
                {'label': 'High', 'value': 3},
                {'label': 'Very High', 'value': 4}
            ],
            value=0
        ),

        html.Label("Extracurricular Activities:"),
        dcc.Dropdown(id='extracurricular', options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], value=0),

        html.Label("Sports:"),
        dcc.Dropdown(id='sports', options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], value=0),

        html.Label("Music:"),
        dcc.Dropdown(id='music', options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], value=0),

        html.Label("Volunteering:"),
        dcc.Dropdown(id='volunteering', options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], value=0),

        html.Br(),
        html.Button('Predict Grade', id='predict-btn'),
        html.Div(id='output')
    ])
])



@app.callback(
    Output('output', 'children'),
    [Input('predict-btn', 'n_clicks'),
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
     Input('volunteering', 'value')]
)
def update_output(n_clicks, age, gender, parental_education, study_time, absences, tutoring, parental_support, extracurricular, sports, music, volunteering):
    ctx = callback_context

    # If the trigger is not the button, clear the output
    if not ctx.triggered or ctx.triggered[0]['prop_id'].split('.')[0] != 'predict-btn':
        return ""

    # === Prediction logic ===
    try:
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

        input_array = np.array([input_vector])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]

        grade_mapping = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}
        predicted_grade = grade_mapping.get(int(prediction), "Unknown")

        grade_color = {
            "A": "#28a745", "B": "#17a2b8", "C": "#ffc107", "D": "#fd7e14", "F": "#dc3545"
        }

        grade_color_style = {
            'color': 'white',
            'backgroundColor': grade_color.get(predicted_grade, '#6c757d'),
            'padding': '10px',
            'borderRadius': '10px',
            'display': 'inline-block',
            'fontSize': '24px',
            'fontWeight': 'bold',
            'marginBottom': '10px'
        }

        if support_score < 0.3:
            risk_level = ("High Risk", "#dc3545")
        elif support_score < 0.6:
            risk_level = ("Moderate Risk", "#ffc107")
        else:
            risk_level = ("Low Risk", "#28a745")

        risk_badge = html.Div(risk_level[0], style={
            'color': 'white',
            'backgroundColor': risk_level[1],
            'padding': '5px 12px',
            'borderRadius': '20px',
            'display': 'inline-block',
            'fontSize': '16px',
            'marginTop': '10px'
        })

        progress_bar = html.Div([
            html.Label("Support Score", style={'marginTop': '15px'}),
            html.Div(style={
                'backgroundColor': '#e9ecef',
                'borderRadius': '10px',
                'overflow': 'hidden',
                'height': '25px'
            }, children=html.Div(style={
                'width': f"{int(support_score * 100)}%",
                'backgroundColor': '#4CAF50',
                'height': '100%',
                'textAlign': 'center',
                'color': 'white',
                'lineHeight': '25px'
            }, children=f"{int(support_score * 100)}%"))
        ])

        result = [
            html.Div(f"Predicted Grade Class: {predicted_grade}", style=grade_color_style),
            html.Br(),
            risk_badge,
            html.Br(),
            progress_bar
        ]

          # === Expanded Recommendations ===
        recommendations = []

        if study_time < 5:
            recommendations.append("Increase your weekly study time to at least 5 hours to improve academic performance. Students who study less than this tend to struggle with grasping core concepts.")
        if study_time > 15:
            recommendations.append("High study time detected. While it's great that you're dedicated, ensure a balanced approach to avoid burnout. Consider taking breaks and engaging in other activities to refresh your mind.")
        if absences > 20:
            recommendations.append("Critical: Excessive absences. Immediate intervention is recommended. Absenteeism can have a serious impact on your academic performance, and it's important to address this now to avoid further decline.")
        elif absences > 10:
            recommendations.append("Reduce absences to improve consistency in learning. Missing classes frequently disrupts the flow of information and may result in gaps in understanding.")
        if tutoring == 0 and prediction >= 2:
            recommendations.append("Consider joining a tutoring program, especially if your predicted grade is lower than expected. Additional help outside of class can be a key factor in improving grades.")
        if tutoring == 1 and prediction >= 2:
            recommendations.append("Tutoring is active, but grades remain low. It might be beneficial to reassess the current tutoring methods or explore alternative resources, such as peer study groups or additional academic support.")
        if parental_support <= 1:
            recommendations.append("Seek more parental or community academic support. Strong parental involvement is crucial to academic success. Consider discussing strategies with your family or seeking external mentoring or counseling support.")
        elif parental_support >= 3 and prediction >= 2:
            recommendations.append("Despite strong parental support, grades are low. This could indicate underlying learning challenges that may need to be addressed. Consider consulting with an educational psychologist for further assessment.")
        if activity_score == 0:
            recommendations.append("Join extracurriculars to boost engagement and personal development. Activities like sports, music, or volunteering help build valuable skills that can positively impact academic performance and overall well-being.")
        if music == 0 and volunteering == 0:
            recommendations.append("Participate in music or community service for personal development. These activities not only enhance your resume but also provide opportunities for emotional growth and stress relief.")
        if parental_education >= 3 and prediction >= 3:
            recommendations.append("Academic gap alert: Your parents have high educational qualifications, but your grades are below expectations. This could indicate a mismatch between academic expectations and support. Investigate learning styles or alternative teaching methods.")

        if recommendations:
            result.append(html.Br())
            result.append(html.H4("Recommendations:"))
            result.append(html.Ul([html.Li(rec) for rec in recommendations]))
        else:
            result.append(html.Br())
            result.append(html.Div("No critical interventions recommended at this time."))

        return html.Div(result)

    except Exception as e:
        return f"Error during prediction: {str(e)}"

        
# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=True, host="0.0.0.0", port=port)