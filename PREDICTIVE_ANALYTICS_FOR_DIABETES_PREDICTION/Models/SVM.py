import pandas as pd
from sklearn import svm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, matthews_corrcoef, confusion_matrix, \
    accuracy_score, classification_report, f1_score, recall_score, precision_score
file_name= 'diabetes_data_upload.csv'

df= pd.read_csv(file_name)
df=df[['Gender',	'Polyuria',	'Polydipsia',	'sudden weight loss',	'weakness' ,
                                      'Polyphagia','Genital thrush','visual blurring','Itching','Irritability',
                                      'delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity','class']]

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
y=df['class']
# print(X)
# print(y)

from sklearn.preprocessing import StandardScaler, LabelEncoder

for column in X:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

# Encode the target variable
y = LabelEncoder().fit_transform(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

# Making predictions on the test set

y_pred = clf.predict(X_test)
# Create confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Evaluating the classifier accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy:', accuracy)

report = classification_report(y_test, y_pred)

# Calculate metrics
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label=1)  # Assuming 1 is the positive class
mcc = matthews_corrcoef(y_test, y_pred)
tpr = recall_score(y_test, y_pred, pos_label=1)  # True Positive Rate (Sensitivity)
spc = recall_score(y_test, y_pred, pos_label=0)  # Specificity (recall for negative class)
ppv = precision_score(y_test, y_pred, pos_label=1)  # Positive Predictive Value (Precision)

# Generate classification report
class_report = classification_report(y_test, y_pred)

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