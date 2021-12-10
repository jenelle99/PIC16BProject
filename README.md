# Demonstration

Here, we will briefly demonstrate our final project.

## Import necessary packages

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from pywebio.output import *
from pywebio.input import * 
```

## Data Acquisition
We will use the 2019 Mental Health Client-Level Data (MH-CLD) from SAMHDA. You can find the link in the *data* folder.

```python
r = urlopen('https://www.datafiles.samhsa.gov/sites/default/files/MH-CLD-2019-DS0001-bndl-data-sas.zip').read()
file = ZipFile(BytesIO(r)) #read the above URL of a zipped folder
file_ = file.open('mhcld_puf_2019.sas7bdat') #open the relevant file from zipped folder
#drop irrelevant columns and replace the value -9.0 (which is used to encode NaN) by NaN.
df = pd.read_sas(file_, format='sas7bdat').drop(['YEAR', 'NUMMHS', 'MH3', 'CASEID', 'DIVISION', 'REGION', 'SPHSERVICE',
                                                 'CMPSERVICE', 'OPISERVICE', 'RTCSERVICE', 'IJSSERVICE', 'TRAUSTREFLG',
                                                 'ANXIETYFLG', 'ADHDFLG','CONDUCTFLG', 'DELIRDEMFLG', 'BIPOLARFLG',
                                                 'ODDFLG', 'PDDFLG', 'PERSONFLG','SCHIZOFLG', 'ALCSUBFLG',
                                                 'OTHERDISFLG'], axis = 1).replace(-9.0, np.nan)
```

## Data Preparation
In this step, we will prepare out data **df** to be used for our model.

```python
#drop NaN values in all columns except MH1, MH2, DETNLF, DEPRESSFLG and SUB
df = df.dropna(subset=['AGE', 'EDUC', 'ETHNIC', 'RACE', 'GENDER', 'MARSTAT', 'SMISED',
                       'EMPLOY', 'VETERAN', 'LIVARAG', 'SAP', 'STATEFIP'])
#EMPLOY has unemployed option set to the numerical value 5. Here we replace unemployment
#with detailed information from DETNLF, then we drop the DETNLF column.
df['EMPLOY'][df['EMPLOY'] > 4] = df['EMPLOY'] + df['DETNLF'].replace(np.nan, 0.0) - 1
df = df.drop('DETNLF', axis = 1)
#Find the rows where only one mental health disorder diagnosis was given
df = df[df['MH2'].isna()].drop(['MH2'], axis = 1)

#create a dataset for deecting depressive disorders
#(we don't need MH1 here because we know it's depressive disorder)
dfdep = df.drop(['MH1'], axis = 1).dropna(subset = ['DEPRESSFLG'])
dfdep = dfdep.replace(np.nan, 14) #replace nan in substance use to be 14:None

#Find the rows where mental illness is NOT depressive disorder
df = df[df['MH1']!=7].drop(['DEPRESSFLG'], axis=1).dropna(subset = ['MH1'])
#Encode labels so that evevry mental illness with key > 7 is shifted down by 1
df['MH1'][df['MH1'] > 6] = df['MH1'][df['MH1'] > 6] - 1
df = df.replace(np.nan, 14) #replace nan in substance use to be 14:None
```

## Data Preprocessing and Subset
We will create a function `clean(y)` to make the target `y` have 50% of label 0 and the rest label 1.

```python
def clean(y):
    """
    This function randomly selects rows from y where the label is 0. The size
    of the randomly slected rows is the same as the number of rows with label 1
    """
    dataf = y[y == 1].append(y[y==0].sample(len(y[y == 1])))
    return dataf.sample(frac=1) #shuffle results
```

Then, we will divide the data into **training** and **testing** set.
```python
# first model training and testing data
y = df['MH1']-1 #reencode the target variable to start at 0
X = df.drop(['MH1'], axis = 1) #training dataset
#split into training and testing data with 20% of the data used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# second model training and testing data
ydep = clean(dfdep['DEPRESSFLG']) #traget variable is already in binary
dfdep = dfdep.loc[ydep.index] #change the dataframe according to slected rows in y
Xdep = dfdep.drop(['DEPRESSFLG'], axis = 1) #depressive disorder training dataset
#split into training and testing data with 20% of the data used for testing
X_traind, X_testd, y_traind, y_testd = train_test_split(Xdep, ydep, test_size = 0.2)
```

## Two Models
We will now create two models.

```python
#create 1st model
model = models.Sequential()
model.add(layers.Dense(15, input_dim = 13, activation = 'relu'))
model.add(layers.Dense(15, activation = 'relu'))
model.add(layers.Dense(12, activation = 'softmax'))
```

The first model is for predicting possible mental illnesses (other than depressive disorders). This model should have 12 output classes that correspond to the 12 different mental illnesses.

```python
#create 2nd model
model2 = models.Sequential()
model2.add(layers.Dense(32, input_dim = 13, activation='relu'))
model2.add(layers.Dense(1, activation='sigmoid'))
```

The second is for detecting depressive disorders. This model will output `1` if depressive disorder is detected and `0` if not. We use Tensorflow to perform machine learning.

## First Model Performance
```python
#compile 1st model with 20 epochs
model.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])
history = model.fit(X_train,
                    y_train,
                    epochs = 20,
                    verbose = 0,
                    validation_data = (X_test, y_test))
#display final model accuracy, validation accuracy, and the baseline rate
print('Model accuracy: ' + str(history.history.get('accuracy')[-1])
      + ', Validation accuracy: ' + str(history.history.get('val_accuracy')[-1])
      + ', Baseline rate: ' + str(len(df[df['MH1'] == df.MH1.mode()[0]])/len(df)))
```

This base model has performance of:

**Model accuracy: 0.4047476351261139, Validation accuracy: 0.40914642810821533, Baseline rate: 0.22609908667058315**

```python
#compile 2nd model with 20 epochs
model2.compile(optimizer='adam',
                  loss= 'binary_crossentropy',
                  metrics=['accuracy'])
history2 = model2.fit(X_traind,
                    y_traind,
                    epochs = 20,
                    verbose = 0,
                    validation_data = (X_testd, y_testd))
#display final model accuracy, validation accuracy, and the baseline rate
print('Model accuracy: ' + str(history2.history.get('accuracy')[-1])
      + ', Validation accuracy: ' + str(history2.history.get('val_accuracy')[-1])
      + ', Baseline rate: ' + str(len(dfdep[dfdep['DEPRESSFLG'] == dfdep.DEPRESSFLG.mode()[0]])/len(dfdep)))
```

**Model accuracy: 0.6275374889373779, Validation accuracy: 0.6302738785743713, Baseline rate: 0.5**

## Front-End Work: Webapp
First, we will need dictionaries to translate our variables into more understandable values.
This `invert()` function will swap keys with values in a dictionary

```python
def invert(dictionary):
    """
    This function swaps keys with values in a dictionary
    """
    return {v: k for k, v in dictionary.items()}
```

We will simply create dictionaries:

```python
#define dictionaries
AGE = {1 : "0-11 years", 2 : "12-14 years", 3 : "15-17 years", 4 : "18-20 years",
       5 : "21-24 years", 6 : "25-29 years", 7 : "30-34 years", 8 : "35-39 years",
       9 : "40-44 years", 10 : "45-49 years", 11 : "50-54 years",
       12 : "55-59 years", 13 : "60-64 years", 14 : "65 years and older"}
AGE = invert(AGE)
EDUC = {1 : "Special education", 2 : "0 to 8", 3 :"9 to 11", 4 :"12 (or GED)",
        5 :"More than 12"}
EDUC = invert(EDUC)
ETHNIC = {1 : "Mexican", 2 : "Puerto Rican",
          3 : "Other Hispanic or Latino origin",
          4 : "Not of Hispanic or Latino origin"}
ETHNIC = invert(ETHNIC)
RACE = {1 : "American Indian/Alaska Native", 2 : "Asian",
        3 : "Black or African American",
        4 : "Native Hawaiian or Other Pacific Islander", 5 : "White",
        6 : "Some other race alone/two or more races "}
RACE = invert(RACE)
GENDER = {1 : "Male", 2 : "Female"}
GENDER = invert(GENDER)
SAP = {1 :"Yes", 2 :"No"}
SAP = invert(SAP)
SUB = {1  : "Alcohol-induced disorder", 2  : "Alcohol intoxication",
       3  : "Substance-induced disorder", 4  : "Alcohol dependence",
       5  : "Cocaine dependence", 6  : "Cannabis dependence",
       7  : "Opioid dependence", 8  : "Other substance dependence",
       9  : "Alcohol abuse", 10 : "Cocaine abuse", 11 : "Cannabis abuse",
       12 : "Opioid abuse", 13 : "Other substance related conditions",
       14 : "None"}
SUB = invert(SUB)
MARSTAT = {1 : "Never married", 2 : "Now married", 3 : "Separated",
           4 : "Divorced, widowed"}
MARSTAT = invert(MARSTAT)
SMISED = {1 : "SMI", 2 : "SED and/or at risk for SED", 3 : "Not SMI/SED"}
SMISED = invert(SMISED)
EMPLOY = {1 : "Full-time", 2 : "Part-time",
          3 : "Employed full-time/part-time not differentiated",
          4 : "Unemployed", 5 : "Retired, disabled", 6 : "Student",
          7 : "Homemaker", 8 : "Sheltered/non-competitive employment",
          9 : "Unemployed(Other)"}
EMPLOY = invert(EMPLOY)
VETERAN = {1 :"Yes", 2 :"No"}
VETERAN = invert(VETERAN)
LIVARAG = {1 : "Homeless", 2 : "Private residence", 3 : "Other"}
LIVARAG = invert(LIVARAG)
STATEFIP = {1 :'Alabama', 2 :'Alaska', 4 :'Arizona', 5 :'Arkansas',
            6 :'California', 8 :'Colorado', 9 :'Connecticut', 10 :'Delaware',
            11 :'District of Columbia', 12 :'Florida', 13 :'Georgia',
            15 :'Hawaii', 16 :'Idaho', 17 :'Illinois', 18 :'Indiana',
            19 :'Iowa', 20 :'Kansas', 21 :'Kentucky', 22 :'Louisiana',
            23 :'Maine', 24 :'Maryland', 25 :'Massachusetts', 26 :'Michigan',
            27 :'Minnesota', 28 :'Mississippi', 29 :'Missouri', 30 :'Montana',
            31 :'Nebraska', 32 :'Nevada', 33 :'New Hampshire', 34 :'New Jersey',
            35 :'New Mexico', 36 :'New York', 37 :'North Carolina',
            38 :'North Dakota', 39 :'Ohio', 40 :'Oklahoma', 41 :'Oregon',
            42 :'Pennsylvania', 44 :'Rhode Island', 45 :'South Carolina',
            46 :'South Dakota', 47 :'Tennessee', 48 :'Texas', 49 :'Utah',
            50 :'Vermont', 51 :'Virginia', 53 :'Washington',
            54 :'West Virginia', 55 :'Wisconsin', 56 :'Wyoming',
            72 :'Puerto Rico', 99 :'Other jurisdictions'}
STATEFIP = invert(STATEFIP)
MH = {1 : 'Trauma- and stressor-related disorders',
      2 : 'Anxiety disorders',
      3 : 'Attention deficit/hyperactivity disorder (ADHD)',
      4 : 'Conduct disorders',
      5 : 'Delirium, dementia',
      6 : 'Bipolar disorders',
      7 : 'Oppositional defiant disorders',
      8 : 'Pervasive developmental disorders',
      9 : 'Personality disorders',
      10 : 'Schizophrenia or other psychotic disorders',
      11 : 'Alcohol or substance use disorders',
      12 : 'Other disorders/conditions'}
```
## Create high-end script using PyWebIO
For our webapp we use PyWebIO which we found to be best to obtain user input and output content on the browser. The webapp displays a survey which asks the user to select the most appropriate value for each question from the dropdown menu. After completing the survey, the user can click submit to see our model's predictions. Running this code block will open a new tab in the browser with our WebApp in in.

```python
#WebApp header
put_markdown('# Mental Health Diagnosis Survey').style('color: MediumSeaGreen; font-style: bold;')

#user slections. We use the above-defined dictionaries to display drop down menues.
selections = input_group('Please answer the following questions to help us make a mental health diagnosis prediction for you',
                         [select(label = 'Age Group', options = AGE.keys(), name = 'age'),
                          select(label = 'Education', options = EDUC.keys(), name = 'education'),
                          select(label = 'Ethnicity', options = ETHNIC.keys(), name = 'ethnicity'),
                          select(label = 'Race', options = RACE.keys(), name = 'race'),
                          select(label = 'Gender', options = GENDER.keys(), name = 'gender'),
                          select(label = 'Substance use diagnosis', options = SUB.keys(), name = 'sub'),
                          select(label = 'Marital status', options = MARSTAT.keys(), name = 'marriage'),
                          select(label = 'Do you have serious mental illness (SMI) or serious emotional disturbance (SED)?', options = SMISED.keys(), name = 'smi'),
                          select(label = 'Substance use problem', options = SAP.keys(), name = 'sap'),
                          select(label = 'Employment status', options = EMPLOY.keys(), name = 'employment'),
                          select(label = 'Veteran status', options = VETERAN.keys(), name = 'veteran'),
                          select(label = 'Residential status', options = LIVARAG.keys(), name = 'res'),
                          select(label = 'State', options = STATEFIP.keys(), name = 'state')])

#ask the user to wait for output
put_text("Please wait until we make our predictions!")
#display note about our model and webapp
put_text("Please note that this webapp uses machine learning to make prediction of possible mental illnesses that a person might suffer from. The model is built using SAMHDA data from the year 2019.\nWhile we hope that this app will help users know more about mental health illnesses that they might suffer from, we strongly recommend seeking professional diagosis when experiencing any symptomes.").style('color:DarkBlue; font-style: italic;')
#display waiting bar
put_processbar('bar')
for i in range(1, 11):
    import time
    set_processbar('bar', i / 10)
    time.sleep(0.1)
    
#get user's input into an array
input_ = np.array([[AGE[selections['age']], EDUC[selections['education']], ETHNIC[selections['ethnicity']],
                    RACE[selections['race']], GENDER[selections['gender']],SUB[selections['sub']],
                    MARSTAT[selections['marriage']], SMISED[selections['smi']], SAP[selections['sap']],
                    EMPLOY[selections['employment']], VETERAN[selections['veteran']], LIVARAG[selections['res']],
                    STATEFIP[selections['state']]]])
#make model1 predictions
q1 = model.predict(input_)
prediction1 = np.argmax(q1, axis=1)[0] + 1
#make model2 predictions
q2 = model2.predict(input_)
prediction2 = round(q2[0][0])

#display results
put_text('A mental disorder that you may have is: ')
put_text(MH[prediction1])
if prediction2 == 1: #show this only if model predicts a depressive disorder
    put_text('You might also have depressive disorders')
```

## Final WebApp

Here's a gif showing how to use the app, and an example of results.

![Alt Text](/images/demo_gif.gif)
