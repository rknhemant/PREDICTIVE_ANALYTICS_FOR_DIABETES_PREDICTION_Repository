# Import necessary libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, matthews_corrcoef

from sklearn.metrics import recall_score, precision_score

# Load the dataset
file_path = 'diabetes_data_upload.csv'
df = pd.read_csv(file_path)

# Encode categorical variables
df['Gender'] = df["Gender"].replace({"Female": 1, "Male": 0})
df['Polyuria'] = df['Polyuria'].replace({"Yes": 1, "No": 0})
df['Polydipsia'] = df['Polydipsia'].replace({"Yes": 1, "No": 0})
df['sudden weight loss'] = df['sudden weight loss'].replace({"Yes": 1, "No": 0})
df['weakness'] = df['weakness'].replace({"Yes": 1, "No": 0})
df['Polyphagia'] = df['Polyphagia'].replace({"Yes": 1, "No": 0})
df['Genital thrush'] = df['Genital thrush'].replace({"Yes": 1, "No": 0})
df['visual blurring'] = df['visual blurring'].replace({"Yes": 1, "No": 0})
df['Itching'] = df['Itching'].replace({"Yes": 1, "No": 0})
df['Irritability'] = df['Irritability'].replace({"Yes": 1, "No": 0})
df['delayed healing'] = df['delayed healing'].replace({"Yes": 1, "No": 0})
df['partial paresis'] = df['partial paresis'].replace({"Yes": 1, "No": 0})
df['muscle stiffness'] = df['muscle stiffness'].replace({"Yes": 1, "No": 0})
df['Alopecia'] = df['Alopecia'].replace({"Yes": 1, "No": 0})
df['Obesity'] = df['Obesity'].replace({"Yes": 1, "No": 0})
df['class'] = df['class'].replace({"Positive": 1, "Negative": 0})

df=df.drop(['Itching','muscle stiffness','Alopecia','Obesity'],axis=1)

# Define features and target variable
X = df.drop("class", axis=1)
y = df["class"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# === Before any preprocessing ===
# Initialize and train GaussianNB
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Make predictions
y_pred = nb_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# print("=== Before Preprocessing ===")
# print("Accuracy:", accuracy)
# print("Confusion Matrix:\n", conf_matrix)
# print("Classification Report:\n", class_report)

# === After Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Re-train GaussianNB on scaled data
nb_model.fit(X_train_scaled, y_train)
y_pred_scaled = nb_model.predict(X_test_scaled)

# Evaluate performance
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
conf_matrix_scaled = confusion_matrix(y_test, y_pred_scaled)
class_report_scaled = classification_report(y_test, y_pred_scaled)

# print("\n=== After Scaling ===")
# print("Accuracy:", accuracy_scaled)
# print("Confusion Matrix:\n", conf_matrix_scaled)
# print("Classification Report:\n", class_report_scaled)


# Calculate various metrics

accuracy = accuracy_score(y_test, y_pred_scaled)

f1 = f1_score(y_test, y_pred_scaled, pos_label=y.unique()[0])  # Assuming first label as positive

mcc = matthews_corrcoef(y_test, y_pred_scaled)

tpr = recall_score(y_test, y_pred_scaled, pos_label=y.unique()[0])  # True Positive Rate (Sensitivity)

spc = recall_score(y_test, y_pred_scaled, pos_label=y.unique()[1])  # Specificity (recall for negative class)

ppv = precision_score(y_test, y_pred_scaled, pos_label=y.unique()[0])  # Positive Predictive Value (Precision)



# Generate classification report

class_report = classification_report(y_test, y_pred_scaled)

# Display all results

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
