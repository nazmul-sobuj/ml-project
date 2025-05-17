import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')  # Optional landing page

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  # Form page

    else:
        try:
            # Get form data
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing'))
            )

            # Convert to DataFrame
            pred_df = data.get_data_as_dataframe()

            # Load prediction pipeline and predict
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            return render_template('home.html', results=round(results[0], 2))

        except Exception as e:
            return f"An error occurred: {e}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
