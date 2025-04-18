# Student Grade Class Predictor

A web application powered by machine learning (Random Forest) that predicts the **grade class** (A–F) of a student based on various academic and socio-behavioral features. The application also provides intelligent risk assessments and actionable recommendations to help improve academic performance.

---

## Features

- **Predicts student grade class (A, B, C, D, F)**
- **Considers multiple student metrics**: age, study time, absences, parental support, activities, etc.
- **Calculates risk level** based on support score
- **Visual feedback** via color-coded grade prediction
- **Contextual recommendations** based on input data
- Fully interactive **Dash web app** for real-time analysis

---

## How It Works

The application uses a trained machine learning model to classify students into one of five grade classes. It uses both raw and engineered features, including:

- Study time
- Absences
- Tutoring support
- Parental education level
- Gender
- Participation in extracurricular activities
- Risk flags and support metrics

The model was trained on a dataset (real or synthetic) and saved using `joblib` or Keras, depending on the architecture. Various versions of the model were tested and stored.

---

## Technologies Used

- **Python**
- **Dash (Plotly)** – Web application framework
- **Scikit-learn** – Model training & preprocessing
- **Keras / TensorFlow** – For deep learning experiments
- **NumPy / Pandas**
- **Joblib** – Model serialization

---

## Project Structure

```
├── artifacts/                    
│   ├── best_model.pkl           # Final selected model (used in production)
│   ├── model_l .pkl             # Alternate ML model (version 1)
│   ├── model_2.pkl              # Alternate ML model (version 2)
│   ├── neural_network.keras     # Saved neural network model
│   └── nn_scaler.pkl            # Scaler used with neural network model
├── data/
│   └── Student_performance_data.csv
├── notebooks/
│   └── Guided_project.ipynb     # Full EDA and model training notebook
├── src/
│   ├── assets/
│   │   └── styles.css           # Custom CSS styles
│   └── web_app.py               # Main Dash application
├── .gitignore
├── ReadMe.md
└── requirements.txt
```

> Note: `best_model.pkl` is the primary file used by `web_app.py`.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/KGS577999/Guided-Project
cd Guided-Project
```

### 2. (Optional) Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
cd "v src"
python web_app.py
```

Open http://localhost:8050 in your browser.

---

## Model Details

- **Model Type**: (Linear Regression, Random Forest, XGBoost and Neural Network)
- **Input Features**: 22 total features (raw + engineered)
- **Scaling**: Preprocessing applied using `StandardScaler` or `MinMaxScaler`
- **Output**: Discrete grade class: A, B, C, D, or F

Key engineered features:
- `support_score`
- `at_risk`
- `activity_score`
- `absence_study_interaction`
- `support_per_activity`
- `tutoring_study_interaction`
- `multiple_activities`

---

## Input Fields (UI)

| Feature                    | Type     | Example       |
|---------------------------|----------|---------------|
| Age                       | Numeric  | 18            |
| Gender                    | Categorical | Male / Female |
| Parental Education Level  | Ordinal  | 0 - 4         |
| Weekly Study Time (hrs)   | Numeric  | 6             |
| Absences                  | Numeric  | 3             |
| Tutoring                  | Binary   | Yes / No      |
| Parental Support          | Ordinal  | 0 - 4         |
| Extracurricular           | Binary   | Yes / No      |
| Sports                    | Binary   | Yes / No      |
| Music                     | Binary   | Yes / No      |
| Volunteering              | Binary   | Yes / No      |

---

## Risk Assessment Logic

The app calculates a **Support Score** using:
```python
support_score = 0.45 * tutoring + 0.1 * parental_education + 0.45 * (parental_support / 4)
```

Risk categories:
- **Low Risk**: Score ≥ 0.6
- **Moderate Risk**: 0.3 ≤ Score < 0.6
- **High Risk**: Score < 0.3

This is shown using color-coded badges and progress bars in the app.

---

## Recommendations System

Based on model output and input conditions, the app provides targeted academic advice:
- Study time too low or too high
- Excessive absences
- Missing tutoring or support
- Lack of extracurriculars
- Mismatched parental education vs student performance

These suggestions appear automatically after prediction.

---

## Deployment Notes

To deploy on platforms like Heroku or Render:

- Add a `Procfile`:
  ```
  web: python web_app.py
  ```

- Ensure your port is dynamic:
  ```python
  port = int(os.environ.get("PORT", 8050))
  app.run(host="0.0.0.0", port=port)
  ```

---

## Contributors

This project was collaboratively developed by:

- **M.C. Wolmarans**
- **Caitlin Burnett**
- **Kyle Smith**
- **Paul-Dieter Brandt**

---

## Future Improvements

- Add login and user tracking
- Integrate real student data
- Improve UI responsiveness
- Export personalized reports as PDF
