import pandas as pd
import joblib
import pickle
import gzip

mypipeline = joblib.load('/Users/HomeMac/Documents/Centennial College/Semester 2/Projects/COMP247/9. Project/Submission 2/models/pipeline.sav')
model = joblib.load('/Users/HomeMac/Documents/Centennial College/Semester 2/Projects/COMP247/9. Project/Submission 2/models/logistic_model.sav')

d = {'Occurrence_Year': 2019, 'Occurrence_Month': 'April', 'Report_Year': 2019, 'Report_Month': 'April', 'Division':'D14', 'Hood_ID':'84', 'Premises_Type':'Apartment', 'Bike_Type':'EL', 'Bike_Colour':'BLK   ', 'Cost_of_Bike':200}

df = pd.DataFrame.from_dict([d])
transformed_input = mypipeline.transform(df)

a = model.predict(transformed_input)