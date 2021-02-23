# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 19:54:53 2019

@author: Schnitzel
"""

# ****************************** Preprocessing ******************************
import pandas as pd 
import numpy as np 

# Import data 
data = pd.read_csv('clinical_dataset.csv', sep=';', index_col=False)

# Replace erroneous values 
data.replace([999,"test non realizable","Test not adequate"],[np.nan,np.nan,np.nan], inplace=True)

# Find # of empty values for each attribute
empty_vals_table = pd.concat([data.isnull().sum(), 100 * data.isnull().sum() / len(data)], axis=1, sort=False)
empty_vals_table.columns = ['# of empty values', '% of empty values' ]
# Find # of empty entries for each participant
empty_part_table = pd.concat([data.isna().sum(axis=1), 100*data.isna().sum(axis=1)/55],axis=1, sort=False)
empty_part_table.columns = ['# of empty values', '% of empty values' ]
 
# Remove participants with more than 10 missing entries                            
data = data.dropna(thresh = 45) 
                            
# Find nominal data 
nomin_data = data.select_dtypes(include=['object','bool_']).copy()

# Chi-Square Test
from scipy.stats import chi2_contingency
from scipy.stats import chi2
names = list()
for i in range(1,len(nomin_data.columns)):    
    variab = nomin_data.columns[i]
    table = pd.crosstab(nomin_data['fried'],nomin_data.iloc[:,i]) # contingency table
    stat, p, dof, expected = chi2_contingency(table)	
    # interpret test-statistic
    prob = 0.95
    critical = chi2.ppf(prob, dof)
    if abs(stat) >= critical:
       print('Variables fraid and ',variab, 'are: Dependent')
    else:
       print('Variables fraid and ',variab, 'are: Independent')
       names.append(variab)
       
# Convert nominal features to numerical
cleanup_nums = {"fried": {"Non frail": 0, "Pre-frail": 1, "Frail": 2},
                "gender": {"F":1, "M":0},
                "ortho_hypotension": {"Yes": 1, "No": 0},
                "vision": {"Sees poorly":0, "Sees moderately": 1, "Sees well": 2},
                "audition": {"Hears poorly": 0, "Hears moderately": 1, "Hears well": 2},
                "weight_loss": {"Yes":1,"No":0},
                "balance_single": {">5 sec": 1, "<5 sec": 0},
                "gait_speed_slower": {"Yes":1, "No":0},
                "grip_strength_abnormal": {"Yes":1, "No":0},
                "low_physical_activity":{"Yes":1, "No":0},
                "memory_complain": {"Yes":1, "No":0},
                "sleep": {"No sleep problem":2, "Occasional sleep problem":1, "Permanent sleep problem":0},
                "living_alone": {"Yes":1, "No":0},
                "leisure_club": {"Yes":1, "No":0},
                "house_suitable_participant": {"Yes":1, "No":0},
                "house_suitable_professional": {"Yes":1, "No":0},
                "health_rate": {"1 - Very bad": 1, "2 - Bad": 2, "3 - Medium":3, "4 - Good":4, "5 - Excellent":5},
                "health_rate_comparison": {"1 - A lot worse": 1, "2 - A little worse": 2, "3 - About the same":3, "4 - A little better":4, "5 - A lot better":5},
                "activity_regular": {"> 5 h per week":3, "No":0, "> 2 h and < 5 h per week":2, "< 2 h per week":1 },
                "smoking": {"Never smoked":0, "Past smoker (stopped at least 6 months)":1, "Current smoker": 2}
                 }         
nomin_data.replace(cleanup_nums, inplace=True)   
nomin_data.replace([True, False],[1,0], inplace=True)

# Handle missing categorical data
nom0 = nomin_data.loc[nomin_data['fried']==0]
for column in nom0.columns:
    nom0[column].fillna(nom0[column].mode()[0], inplace=True) 
nomin_data.update(nom0) 
    
nom1 = nomin_data.loc[nomin_data['fried']==1]
for column in nom1.columns:
    nom1[column].fillna(nom1[column].mode()[0], inplace=True)
nomin_data.update(nom1) 
    
nom2 = nomin_data.loc[nomin_data['fried']==2]
for column in nom2.columns:
    nom2[column].fillna(nom2[column].mode()[0], inplace=True)
nomin_data.update(nom2) 

# Update dataset with the processed categorical data
data.update(nomin_data)

# Handle missing values in numerical data 
numer_data = data.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).copy()

num0 = numer_data.loc[nomin_data['fried']==0]
for column in num0.columns:
    num0[column].fillna(num0[column].mean(), inplace=True) 
numer_data.update(num0) 
    
num1 = numer_data.loc[nomin_data['fried']==1]
for column in num1.columns:
    num1[column].fillna(num1[column].mean(), inplace=True)
numer_data.update(num1) 
    
num2 = numer_data.loc[nomin_data['fried']==2]
for column in numer_data.columns:
    num2[column].fillna(num2[column].mean(), inplace=True)
numer_data.update(num2) 

# Update dataset with the processed categorical data
data.update(numer_data)

#print('Check if any empty values in the data:', data.isnull().sum())

# Visualize the data
#data.hist(bins=54, figsize=(20,20))

# Display correlation pairs in descending order
import itertools
df = pd.DataFrame([[(i,j),data.corr().loc[i,j]] for i,j in list(itertools.combinations(data.corr(), 2))],columns=['pairs','corr'])    
#print(df.sort_values(by='corr',ascending=False))

# Display variance in ascending order
v = data.var()
#print(v.sort_values(ascending=True))

data = data.drop('hospitalization_one_year',axis=1)

# Save final dataset as a csv file
data.to_csv('clinical_final.csv',sep=';',index=False)

# ****************************** Classification ******************************

# Divide data into target (label) and features
labels = data['fried']
labels=labels.astype('int')
features = data.drop(['fried','part_id','weight_loss', 'exhaustion_score', 'gait_speed_slower','grip_strength_abnormal','low_physical_activity'],axis=1)

# Scale data
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
features = scaler.fit_transform(features)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.20)

# Define a function for fitting a classifier and calculating the accuracy score
from sklearn.metrics import accuracy_score

def classification(model, classifier):
    # build the model on training data
    model.fit(train_features, train_labels)
    # make predictions for test data
    predictions = model.predict(test_features)
    # calculate the accuracy score
    accuracy = accuracy_score(test_labels, predictions)*100
    print('Accuracy score of',classifier,'Classifier %0.2f:' % accuracy,'%')
    
# Define a function for performing cross validation on the model
from sklearn.model_selection import cross_val_score

def cv(model,classifier,features,label):
    scores = cross_val_score(model, features, label, cv=10)*100
    print('Cross Validation of',classifier,'model - Accuracy: %0.2f (+/- %0.2f)'  % (scores.mean(), scores.std() * 2),'% \n')

# Apply different algorithms for classification
   
# --------- Random Forest ---------
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 500, max_depth = 4,random_state=0)
classification(model, "Random Forest")
cv(model, "Random Forest",features,labels)    
    
# --------- Decision Tree ---------
from sklearn import tree
model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
classification(model, 'Decision Tree')
cv(model, "Decision Tree", features,labels)

# --------- SVM Classifier ---------
from sklearn.svm import SVC
model = SVC(gamma='auto')
classification(model, "SVM")
cv(model, "SVM",features,labels)

# --------- Gaussian Naive Bayes ---------
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
classification(model, "Gaussian Naive Bayes")
cv(model, "Gaussian Naive Bayes",features,labels)

# --------- Logistic Regression ---------
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=0,max_iter=1000)
classification(model, "Logistic Regression")
cv(model, "Logistic Regression",features,labels)


