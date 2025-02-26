import pandas as pd


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, matthews_corrcoef
from sklearn.metrics import recall_score, precision_score

df= pd.read_csv('diabetes_data_upload.csv')


df["Gender"].replace({"Female":1,"Male":0},inplace=True)
df['Polyuria'].replace({"Yes":1,"No":0},inplace=True)
df['Polydipsia'].replace({"Yes":1,"No":0},inplace=True)
df['sudden weight loss'].replace({"Yes":1,"No":0},inplace=True)
df['weakness'] .replace({"Yes":1,"No":0},inplace=True)
df['Polyphagia'].replace({"Yes":1,"No":0},inplace=True)
df['Genital thrush'].replace({"Yes":1,"No":0},inplace=True)
df['visual blurring'].replace({"Yes":1,"No":0},inplace=True)
df['Itching'].replace({"Yes":1,"No":0},inplace=True)
df['Irritability'].replace({"Yes":1,"No":0},inplace=True)
df['delayed healing'].replace({"Yes":1,"No":0},inplace=True)
df['partial paresis'].replace({"Yes":1,"No":0},inplace=True)
df['muscle stiffness'].replace({"Yes":1,"No":0},inplace=True)
df['Alopecia'].replace({"Yes":1,"No":0},inplace=True)
df['Obesity'].replace({"Yes":1,"No":0},inplace=True)
df['class'].replace({"Positive":1,"Negative":0},inplace=True)

df=df.drop(['Itching','muscle stiffness','Alopecia','Obesity'],axis=1)

x  =df.drop("class",axis=1)
y = df["class"]
x_train,x_test,y_train,y_test  = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=42,stratify=y)
# training model

dt_model = DecisionTreeClassifier()

dt_model.fit(x_train,y_train)


y_pred = dt_model.predict(x_train)
cnf_matrix = confusion_matrix(y_train,y_pred)
accracy = accuracy_score(y_train,y_pred)

classification_rpt = classification_report(y_train,y_pred)

################

y_pred = dt_model.predict(x_test)
cnf_matrix = confusion_matrix(y_test,y_pred)
accracy = accuracy_score(y_test,y_pred)
classification_rpt = classification_report(y_test,y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred, pos_label=1)  # Assuming 1 as positive class label
mcc = matthews_corrcoef(y_test, y_pred)
tpr = recall_score(y_test, y_pred, pos_label=1)  # True Positive Rate (Sensitivity)

spc = recall_score(y_test, y_pred, pos_label=0)  # Specificity (recall for negative class)

ppv = precision_score(y_test, y_pred, pos_label=1)  # Positive Predictive Value (Precision)



# Generate classification report

class_report = classification_report(y_test, y_pred)



# Display all results

print(
    "conf_matrix \n", conf_matrix,
    "\n accuracy = ", round(accuracy, 4),
    "\n f1 = ", round(f1, 4),
    "\n mcc = ", round(mcc, 4),
    "\n tpr = ", round(tpr, 4),
    "\n spc = ", round(spc, 4),
    "\n ppv = ", round(ppv, 4),
    "\n class_report \n", class_report)
print("============================")



