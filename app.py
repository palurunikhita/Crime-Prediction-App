from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import zipfile

with zipfile.ZipFile("dataset/CrimeDataset.csv.zip", 'r') as zipref:
    zipref.extractall("dataset/")

app = Flask(__name__)
baseDir = os.path.abspath(os.path.dirname(__file__))
modelPath = os.path.join(baseDir, 'models', 'model.pkl')
datasetPath = os.path.join(baseDir, 'dataset', 'CrimeDataset.csv')

# Loading the trained model and model name
bestModel = joblib.load(modelPath)
with open('models/modelName.txt') as f:
    modelUsed = f.read()

# Loading the dataset and preparing the data for dropdowns in UI
df = pd.read_csv(datasetPath)

def createMapping(value, ranking):
    sortedDf = df.dropna(subset=[value, ranking]).sort_values(by=ranking)
    uniqueValues = sortedDf[value].unique()
    rankings = sortedDf.drop_duplicates(subset=[value])[ranking].tolist()
    return dict(zip(uniqueValues, rankings))

# Mapping each input category to its ranking
sexMapping = createMapping('Vict Sex', 'Sex-Crime Ranking')
areaMapping = createMapping('Area Name', 'Area-Crime Ranking')
ageMapping = createMapping('Age Brackets', 'Age-Crime Ranking')
timeMapping = createMapping('Time Slots', 'Time-Crime Ranking')

descentMapping = {
    "Other Asian" : "A",
    "Black" : "B",
    "Chinese" : "C",
    "Cambodian" : "D",
    "Filipino" : "F",
    "Guamanian" : "G",
    "Hispanic/Latin/Mexican" : "H",
    "American Indian/Alaskan Native" : "I",
    "Japanese" : "J",
    "Korean" : "K",
    "Laotian" : "L",
    "Other" : "O",
    "Pacific Islander" : "P",
    "Samoan" : "S",
    "Hawaiian" : "U",
    "Vietnamese" : "V",
    "White" : "W",
    "Unknown" : "X",
    "Asian Indian" : "Z"
}

descentReverseMapping = {v:k for k,v in descentMapping.items()}
descentValues = createMapping('Vict Descent', 'Descent-Crime Ranking')

@app.route('/')
def home():
    return render_template('index.html', 
                           uniqueSex=list(sexMapping.keys()), 
                           uniqueDescent=list(descentReverseMapping.values()), 
                           uniqueArea=list(areaMapping.keys()),
                           uniqueAge=list(ageMapping.keys()), 
                           uniqueTime=list(timeMapping.keys()))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.form
        victimSex = sexMapping[data['victim_sex']]
        victimDescent = descentValues[descentMapping[data['victim_descent']]]
        victimAge = ageMapping[data['age_bracket']]
        timeSlot = timeMapping[data['time_slot']]
        areaName = areaMapping[data['area_name']]

        inputFeatures = pd.DataFrame([[victimAge, victimSex, victimDescent, areaName, timeSlot]],
                                     columns=['Age-Crime Ranking','Sex-Crime Ranking', 'Descent-Crime Ranking',
                                              'Area-Crime Ranking','Time-Crime Ranking'])

        prediction = bestModel.predict(inputFeatures)
        crimeLevels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

        if isinstance(prediction[0], str):
            crimeLikelihood = prediction[0]
        else:
            crimeLikelihood = crimeLevels[int(prediction[0])]

        return render_template('index.html', prediction=crimeLikelihood,
                               modelName=modelUsed,
                               uniqueSex=list(sexMapping.keys()), 
                               uniqueDescent=list(descentReverseMapping.values()), 
                               uniqueArea=list(areaMapping.keys()), 
                               uniqueAge=list(ageMapping.keys()), 
                               uniqueTime=list(timeMapping.keys()),
                               method=request.method)
    
    return render_template('index.html',
                           uniqueSex=list(sexMapping.keys()), 
                           uniqueDescent=list(descentReverseMapping.values()), 
                           uniqueArea=list(areaMapping.keys()), 
                           uniqueAge=list(ageMapping.keys()), 
                           uniqueTime=list(timeMapping.keys()),
                           method=request.method)


if __name__ == '__main__':
    app.run(debug=True)