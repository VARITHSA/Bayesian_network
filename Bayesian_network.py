import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from sklearn.preprocessing import LabelEncoder

names = "age,sex,cp,trestbps,chol,fbs,estecg,thalach,exang,oldpeak,lope,ca,thal,result"
names = names.split(",")

data = pd.read_csv('./data/processed.cleveland.csv', names = names, nrows= 100)

print(data.head())
print('\n')
print(data.isnull().sum())
data = data.replace(r'\s+', np.nan, regex=True)
print(data.columns)


# Define the parameters and target variable
# Define the parameters and target variable
parameters = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'estecg', 'thalach', 'exang', 'oldpeak', 'lope', 'ca', 'thal']
target_variable = 'result'

# Create a Bayesian Network model
model = BayesianNetwork()

# Add nodes for each parameter and the target variable
for parameter in parameters:
    model.add_node(parameter)
model.add_node(target_variable)

# Encode categorical columns using LabelEncoder
le = LabelEncoder()
for column in parameters + [target_variable]:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])

# Calculate the correlation coefficients between each parameter and the target variable
correlations = {}
for parameter in parameters:
    correlation = np.corrcoef(data[parameter], data[target_variable])[0, 1]
    correlations[parameter] = correlation

# Create edges between each parameter and the target variable based on the correlation coefficients
for parameter, correlation in correlations.items():
    if correlation > 0.1:  # threshold for creating an edge
        model.add_edge(parameter, target_variable)
        
  
# Check the model
print(model.check_model)
# Print the edges in the Bayesian Network
print("Edges in the Bayesian Network:")
for edge in model.edges():
    print(f"{edge[0]} → {edge[1]}")
print(model.edges())

model.fit(data, estimator= MaximumLikelihoodEstimator)
print(model.get_cpds('result'))


infer = VariableElimination(model)

print('\n Probability of Having Heart disease given evidence of Chest pain: \n')
query1 = infer.query(variables=['result'],evidence= {'cp':3})
print(query1)

print('\n Probability of Having Heart Disease given Evidence of High Cholestrol: \n')
query2 = infer.query(variables = ['result'], evidence = {'chol':250})
print(query2)

print('\n Probability of Having Heart Disease given Evidence of high fbs: \n')
query3 = infer.query(variables = ['result'], evidence = {'fbs':1})
print(query3)