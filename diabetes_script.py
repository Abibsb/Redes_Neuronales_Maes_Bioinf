#librerias necesarias para comenzar a trabajar
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file 
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report 

#carga y exploracion del dataset
data=pd.read_csv('C:\\Users\\amari\\Downloads\\RNA\\diabetes.csv')
print(data.shape)
print(data.dtypes)
data.head()

#Opcional: ver el dataset completo
#def print_full(x):
#    pd.set_option('display.max_rows', len(x))
#    print(x)
#    pd.reset_option('display.max_rows')

#print_full(data)

import matplotlib.pyplot as plt
import seaborn as sns

#correlacion entre todas las variables 
corrmat = data.corr()
f, ax = plt.subplots(figsize=(9, 9))
sns.heatmap(corrmat, vmax=.8, square=True, annot=True);

#exploracion de relacion entre edad, test de glucosa y presencia de diabetes
print("Pacientes mayores a 45 años con diagnóstico positivo de diabetes: ")
print(len(data[(data['Age']>45) & (data['Outcome']==1)]))
adultos_mayora45 = data.loc[data["Age"] > 45]
adultos_mayora45.shape
y=adultos_mayora45['Outcome']
plt.figure(figsize=(4,4))
sns.countplot(y).set(title='Pacientes mayores de 45 años')
plt.show()

print("Pacientes menores a 45 años con diagnóstico positivo de diabetes: ")
print(len(data[(data['Age']<44) & (data['Outcome']==1)]))
adultos_menora45 = data.loc[data["Age"] < 44]
adultos_menora45.shape
y=adultos_menora45['Outcome']
plt.figure(figsize=(4,4))
sns.countplot(y).set(title='Pacientes menores a 45 años')
plt.show

positivos_may45= len(data[(data['Age']>45) & (data['Outcome']==1)])
total_may45= len(data[(data['Age']>45)])
porcentaje_may45 = positivos_may45 / total_may45 *100
print('Porcentaje de positivos mayores de 45 años:') 
print(porcentaje_may45)

positivos_men45= len(data[(data['Age']<44) & (data['Outcome']==1)])
total_men45= len(data[(data['Age']<44)])
porcentaje_men45 = positivos_men45 / total_men45 *100
print('Porcentaje de positivos menores de 45 años:') 
print(porcentaje_men45)

print("pacientes con glucosa en sangre inferior a 140 mg/dL con diagnóstico positivo de diabetes: ")
print(len(data[(data['Glucose']<140) & (data['Outcome']==1)]))
glucosa_menor140 = data.loc[data["Glucose"] < 140]
glucosa_menor140.shape
y=glucosa_menor140['Outcome']
plt.figure(figsize=(4,4))
sns.countplot(y).set(title='Glucosa inferior a 140 mg/dL')
plt.show()

positivos_men140 = len(data[(data['Glucose']<140) & (data['Outcome']==1)])
total_men140 = len(data[(data['Glucose']<140)])
porcentaje_men140 = positivos_men140/total_men140 *100
print('Porcentaje de positivos con glucosa inferior a 140 mg/dL:') 
print(porcentaje_men140)

print("pacientes glucosa en sangre superior a 200 mg/dL con diagnóstico positivo de diabetes: ")
print(len(data[(data['Glucose']>200) & (data['Outcome']==1)]))
print("\n")

print("pacientes glucosa en sangre inferior a 200 mg/dL y superior a 140 mg/dL con diagnóstico positivo de diabetes: ")
print(len(data[(data["Glucose"]>141) & (data['Glucose']<200) & (data['Outcome']==1)]))
glucosa_prediabetes = data.loc[(data["Glucose"]>141) & (data['Glucose']<200)]
glucosa_prediabetes.shape
glucosa_prediabetes.head()
y=glucosa_prediabetes['Outcome']
plt.figure(figsize=(4,4))
sns.countplot(y).set(title='Glucosa entre 141 mg/dL y 199 mg/dL')
plt.show()

positivos_predia = len(data[(data["Glucose"]>141) & (data['Glucose']<200) & (data['Outcome']==1)])
total_predia = len(data[(data["Glucose"]>141) & (data['Glucose']<200)])
porcentaje_predia = positivos_predia/total_predia *100
print('Porcentaje de positivos con glucosa entre 140 mg/dL y 199 mg/dL:') 
print(porcentaje_predia)

#Tratamiento de dataset. Reemplazo de valores NaN y escalado de variables
data.rename(columns={'Outcome': 'target'}, inplace=True) 
print(data.isnull().sum())
data.head()

standardScaler = StandardScaler()
columns_to_scale = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age',]
data[columns_to_scale] = standardScaler.fit_transform(data[columns_to_scale])
data.shape
data = data[np.isfinite(data).all(1)]
data.shape
data.head()

#Division de dataset en subsets de entrenamiento y testeo
y = data['target']
X = data.drop('target',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 0)

#Ploteo del target (Diabetes)
plt.figure(figsize=(6,4))
sns.countplot(y)
plt.show()
print(X_train.shape)
print(X_test.shape)

#MULTILAYER PERCEPTRON (MLP)
from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(hidden_layer_sizes=(30), alpha=0.001, max_iter=150)
model = MLP.fit(X_train, y_train)
MLP_predict = MLP.predict(X_test)
MLP_conf_matrix = confusion_matrix(y_test, MLP_predict)
MLP_acc_score = accuracy_score(y_test, MLP_predict)

#Imprimir la matriz de confusion y el score de exactitud("confussion matrix")
print(MLP_conf_matrix)
print("\n")
print(classification_report(y_test,MLP_predict))
print("Exactitud del clasificados MLP: {:.3f}".format(MLP_acc_score*100),'%\n')

#Optimizacion de parametros de MLP
parameter_space = {
    'hidden_layer_sizes': [(30),],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive','invscaling'],
}
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(MLP, parameter_space, n_jobs=-1, cv=5)
clf.fit(X, y) # X is train samples and y is the corresponding labels

print('Best parameters found:\n', clf.best_params_)

#MULTILAYER PERCEPTRON (MLP) con valores optimizados
MLP_op = MLPClassifier(hidden_layer_sizes=(30), alpha=0.0001, max_iter=150, activation='identity', learning_rate='invscaling',solver='adam')
model = MLP_op.fit(X_train, y_train)
MLP_op_predict = MLP_op.predict(X_test)
MLP_op_conf_matrix = confusion_matrix(y_test, MLP_op_predict)
MLP_op_acc_score = accuracy_score(y_test, MLP_op_predict)


#Imprimir la matriz de confusion y el score de exactitud("confussion matrix")
print(MLP_op_conf_matrix)
print("\n")
print(classification_report(y_test,MLP_op_predict))
print("Exactitud del clasificados MLP: {:.3f}".format(MLP_op_acc_score*100),'%\n')

#Curva ROC
from sklearn.metrics import roc_auc_score, roc_curve
y_test_int = y_test.replace({'Good': 1, 'Bad': 0})
auc_model = roc_auc_score(y_test_int, MLP_predict)
fpr_model, tpr_model, thresholds_model = roc_curve(y_test_int, MLP_predict)
auc_model_op = roc_auc_score(y_test_int, MLP_op_predict)
fpr_model_op, tpr_model_op, thresholds_model_op = roc_curve(y_test_int, MLP_op_predict)
plt.figure(figsize=(10, 5))
plt.plot(fpr_model, tpr_model, label=f'AUC (MLP) = {auc_model:.2f}')
plt.plot(fpr_model_op, tpr_model_op, label=f'AUC (MLP_op) = {auc_model:.2f}')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Baseline')
plt.title('ROC Curve', size=20)
plt.xlabel('False Positive Rate', size=14)
plt.ylabel('True Positive Rate', size=14)
plt.legend();