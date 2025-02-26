import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, matthews_corrcoef, confusion_matrix, \
    accuracy_score, classification_report, f1_score, recall_score, precision_score
file_name= 'diabetes_data_upload.csv'

df= pd.read_csv(file_name)

df=df[['Gender',	'Polyuria',	'Polydipsia',	'sudden weight loss',	'weakness' ,
                                      'Polyphagia','Genital thrush','visual blurring','Itching','Irritability',
                                      'delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity','class']]
print('rows,columns',df.shape)
df['Gender'] = df['Gender'].apply(lambda x: 0 if x=='Male' else 1)
df['Polyuria'] = df['Polyuria'].apply(lambda x: 0 if x=='No' else 1)
df['Polydipsia'] = df['Polydipsia'].apply(lambda x: 0 if x=='No' else 1)
df['sudden weight loss'] = df['sudden weight loss'].apply(lambda x: 0 if x=='No' else 1)
df['weakness'] = df['weakness'].apply(lambda x: 0 if x=='No' else 1)
df['Polyphagia'] = df['Polyphagia'].apply(lambda x: 0 if x=='No' else 1)
df['Genital thrush'] = df['Genital thrush'].apply(lambda x: 0 if x=='No' else 1)
df['visual blurring'] = df['visual blurring'].apply(lambda x: 0 if x=='No' else 1)
df['Itching'] = df['Itching'].apply(lambda x: 0 if x=='No' else 1)
df['Irritability'] = df['Irritability'].apply(lambda x: 0 if x=='No' else 1)
df['delayed healing'] = df['delayed healing'].apply(lambda x: 0 if x=='No' else 1)
df['partial paresis'] = df['partial paresis'].apply(lambda x: 0 if x=='No' else 1)
df['muscle stiffness'] = df['muscle stiffness'].apply(lambda x: 0 if x=='No' else 1)
df['Alopecia'] = df['Alopecia'].apply(lambda x: 0 if x=='No' else 1)
df['Obesity'] = df['Obesity'].apply(lambda x: 0 if x=='No' else 1)


df=df.drop(['Itching','muscle stiffness','Alopecia','Obesity'],axis=1)
X=df.drop(columns=['class'])
feature_names= X.columns.tolist()

Y=df['class']
class_names = ['class']
value =  np.array(X)
target_name = np.array(['class'], order='C')
if len(class_names) > 0:
    if len(class_names) > 1:
        target_name = class_names[np.argmax(value)]
    else:
        target_name = np.array(['class'], order='C')
else:
    target_name = None


X_train, X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, random_state=42)
# Feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Random Forest classifier with entropy criterion
rf_classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)

# Train the Random Forest classifier
rf_classifier.fit(X_train_scaled, Y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(Y_test, y_pred)
#print("Accuracy:", accuracy)
report = classification_report(Y_test, y_pred)

# Create confusion matrix
conf_matrix = confusion_matrix(Y_test, y_pred)

# Visualize one of the decision trees in the Random Forest
plt.figure(figsize=(20,10))
c=np.array(['class'], order='C')

# Calculate metrics
conf_matrix = confusion_matrix(Y_test, y_pred)
accuracy = accuracy_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred, pos_label="Positive")  # Assuming 1 is the positive class
mcc = matthews_corrcoef(Y_test, y_pred)
tpr = recall_score(Y_test, y_pred, pos_label="Positive")  # True Positive Rate (Sensitivity)
spc = recall_score(Y_test, y_pred, pos_label="Negative")  # Specificity (recall for negative class)
ppv = precision_score(Y_test, y_pred, pos_label="Positive")  # Positive Predictive Value (Precision)

# Generate classification report
class_report = classification_report(Y_test, y_pred)

# Display results
print(
    "conf_matrix \n", conf_matrix,
    "\n accuracy = ", round(accuracy, 4),
    "\n f1 = ", round(f1, 4),
    "\n mcc = ", round(mcc, 4),
    "\n tpr = ", round(tpr, 4),
    "\n spc = ", round(spc, 4),
    "\n ppv = ", round(ppv, 4),
    "\n class_report \n", class_report
)
