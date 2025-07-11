from flask import Flask, request, render_template
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
# from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.sql.functions import col, when
from pyspark.sql import Row



app = Flask(__name__,template_folder='./templates')


spark = SparkSession.builder.appName("Telecom Churn Prediction").getOrCreate()
model_path = "./pySparkModel"
model = PipelineModel.load(model_path)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
       
            form_data = {
                'gender': request.form['gender'],
                'SeniorCitizen': int(request.form['SeniorCitizen']),
                'Partner': request.form['Partner'],
                'Dependents': request.form['Dependents'],
                'tenure': int(request.form['tenure']),
                'PhoneService': request.form['PhoneService'],
                'MultipleLines': request.form['MultipleLines'],
                'InternetService': request.form['InternetService'],
                'OnlineSecurity': request.form['OnlineSecurity'],
                'OnlineBackup': request.form['OnlineBackup'],
                'DeviceProtection': request.form['DeviceProtection'],
                'TechSupport': request.form['TechSupport'],
                'StreamingTV': request.form['StreamingTV'],
                'StreamingMovies': request.form['StreamingMovies'],
                'Contract': request.form['Contract'],
                'PaperlessBilling': request.form['PaperlessBilling'],
                'PaymentMethod': request.form['PaymentMethod'],
                'MonthlyCharges': float(request.form['MonthlyCharges']),
                'TotalCharges': float(request.form['TotalCharges'])
            }
            print('came ',form_data)

            example_df = spark.createDataFrame([form_data])
            
            print('came here ')

            example_df = example_df.withColumn('EngagementScore', (col('tenure') * col('MonthlyCharges')) / col('TotalCharges'))
            example_df = example_df.withColumn('HasInternetService', (col('InternetService') != 'No').cast('integer'))
            example_df = example_df.withColumn('TenureTimesMonthlyCharges', col('tenure') * col('MonthlyCharges'))
            example_df = example_df.withColumn("ServiceScore",
                (when(col("OnlineSecurity") == "Yes", 1).otherwise(0) +
                 when(col("OnlineBackup") == "Yes", 1).otherwise(0) +
                 when(col("DeviceProtection") == "Yes", 1).otherwise(0)))
            predictions = model.transform(example_df)
            

            result = predictions.select('prediction').collect()[0]['prediction']
            print("result is ",result)
            
        except Exception as e:
            result = f"Error processing the input: {str(e)}"
            print("error is ",e)

        return render_template('index.html', prediction=result)
    return render_template('index.html',prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
