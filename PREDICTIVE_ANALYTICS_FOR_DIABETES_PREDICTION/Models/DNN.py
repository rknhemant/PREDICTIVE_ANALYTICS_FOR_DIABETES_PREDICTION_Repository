from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, matthews_corrcoef, \
    recall_score, precision_score

# Load the dataset
df = pd.read_csv('diabetes_data_upload.csv')

# Encode categorical variables
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['class'] = label_encoder.fit_transform(df['class'])

# Encode all binary categorical columns
binary_columns = ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush',
                  'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis',
                  'muscle stiffness', 'Alopecia', 'Obesity']
for col in binary_columns:
    df[col] = label_encoder.fit_transform(df[col])
df=df.drop(['Itching','muscle stiffness','Alopecia','Obesity'],axis=1)
# Define features (X) and target (y)
X = df.drop('class', axis=1)
y = df['class']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a simple DNN model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
initial_training = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test), verbose=1)

# Evaluate on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
#print("loss, Accuracy after initial training:",loss, accuracy)

# Make predictions and print the confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report

y_pred = (model.predict(X_test) > 0.5).astype("int32")
#print("Confusion Matrix:")
#print(confusion_matrix(y_test, y_pred))
#print("\nClassification Report:")
#print(classification_report(y_test, y_pred))


#==============================
# Calculate various metrics
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred, pos_label=y.unique()[0])  # Assuming first label as positive

mcc = matthews_corrcoef(y_test, y_pred)

tpr = recall_score(y_test, y_pred, pos_label=y.unique()[0])  # True Positive Rate (Sensitivity)

spc = recall_score(y_test, y_pred, pos_label=y.unique()[1])  # Specificity (recall for negative class)

ppv = precision_score(y_test, y_pred, pos_label=y.unique()[0])  # Positive Predictive Value (Precision)


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


