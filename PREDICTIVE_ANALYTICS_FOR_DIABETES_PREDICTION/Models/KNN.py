import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, matthews_corrcoef, \
    recall_score, precision_score

from sklearn.preprocessing import LabelEncoder, StandardScaler

# Define the dataset as a string for this example
file_name= 'diabetes_data_upload.csv'

# Read the dataset into a DataFrame
from io import StringIO
data = pd.read_csv(file_name)

# Inspect the DataFrame to ensure it's loaded correctly
# print(data.head())
# print(data.info())
label_encoder = LabelEncoder()
# Check for missing values
if data.isnull().values.any():
    print("Data contains missing values. Please handle them before proceeding.")
else:
    print("No missing values found in the data.")
# Encode all binary categorical columns
binary_columns = ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush',
                  'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis',
                  'muscle stiffness', 'Alopecia', 'Obesity']
# Preprocess the data
# Separate features and target variable
for col in binary_columns:
    data[col] = label_encoder.fit_transform(data[col])

#data=data.drop(['Polydipsia','Polyuria'],axis=1)
data=data.drop(['Itching','muscle stiffness','Alopecia','Obesity'],axis=1)
# Define features (X) and target (y)
X = data.drop('class', axis=1)
y = data['class']

# Encode categorical features
categorical_columns = X.select_dtypes(include=['object']).columns

for column in categorical_columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

# Encode the target variable
y = LabelEncoder().fit_transform(y)

# Check if the data is non-empty after preprocessing
if X.shape[0] == 0 or y.shape[0] == 0:
    raise ValueError("The dataset is empty after preprocessing. Check the input data and preprocessing steps.")

# Split the data into training and testing sets
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except ValueError as e:
    print(f"Error during train-test split: {e}")
    print(f"Dataset size: {X.shape[0]} samples")
    raise

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
model = KNeighborsClassifier(5)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
#report = classification_report(y_test, y_pred)

# print(f"Accuracy: {accuracy:.2f}")
# print("Classification Report:")
# print("=====================================")
#print(report)
#==============================
# Calculate various metrics
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred, pos_label=1)  # Assuming first label as positive

mcc = matthews_corrcoef(y_test, y_pred)

tpr = recall_score(y_test, y_pred, pos_label=1)  # True Positive Rate (Sensitivity)

spc = recall_score(y_test, y_pred, pos_label=0)  # Specificity (recall for negative class)

ppv = precision_score(y_test, y_pred, pos_label=1)  # Positive Predictive Value (Precision)


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
