import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, matthews_corrcoef, confusion_matrix, \
    accuracy_score, classification_report, f1_score, recall_score, precision_score
from xgboost import XGBClassifier

file_name= 'diabetes_data_upload.csv'


# Read the dataset into a DataFrame
data = pd.read_csv(file_name)

# Inspect the DataFrame to ensure it's loaded correctly
# print(data.head())
# print(data.info())

# Check for missing values
if data.isnull().values.any():
    print("Data contains missing values. Please handle them before proceeding.")
else:
    print("No missing values found in the data.")

data=data.drop(['Itching','muscle stiffness','Alopecia','Obesity'],axis=1)
# Preprocess the data
# Separate features and target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

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
model = XGBClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
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