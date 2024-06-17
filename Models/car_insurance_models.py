import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from joblib import dump

# Load the dataset
df = pd.read_csv('../Datasets/car_insurance_claim.csv')

# Convert non-numerical columns to numerical
df['INCOME'] = df['INCOME'].replace('[\$,]', '', regex=True).astype(float)
df['HOME_VAL'] = df['HOME_VAL'].replace('[\$,]', '', regex=True).astype(float)
df['BLUEBOOK'] = df['BLUEBOOK'].replace('[\$,]', '', regex=True).astype(float)
df['OLDCLAIM'] = df['OLDCLAIM'].replace('[\$,]', '', regex=True).astype(float)
df['CLM_AMT'] = df['CLM_AMT'].replace('[\$,]', '', regex=True).astype(float)

# Fill missing values
df['INCOME'].fillna(df['INCOME'].mean(), inplace=True)
df['HOME_VAL'].fillna(df['HOME_VAL'].mean(), inplace=True)
df['CAR_AGE'].fillna(df['CAR_AGE'].mean(), inplace=True)
df['YOJ'].fillna(df['YOJ'].mean(), inplace=True)
df['OCCUPATION'].fillna('Unknown', inplace=True)

# Select features and target variable
numerical_columns = ['KIDSDRIV', 'AGE', 'HOMEKIDS', 'YOJ', 'INCOME', 'HOME_VAL',
                     'TRAVTIME', 'BLUEBOOK', 'TIF', 'OLDCLAIM', 'CLM_FREQ',
                     'MVR_PTS', 'CLM_AMT', 'CAR_AGE']
X = df[numerical_columns]
y = df['CLAIM_FLAG']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
    ('scaler', StandardScaler()),  # Scale the data
    ('classifier', MLPClassifier())  # MLP Classifier
])

# Hyperparameters for RandomizedSearchCV
param_dist = {
    'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'classifier__activation': ['relu', 'tanh'],
    'classifier__alpha': [0.0001, 0.001, 0.01],
    'classifier__solver': ['adam', 'sgd'],
    'classifier__learning_rate': ['constant', 'adaptive'],
}

# Define RandomizedSearchCV with Stratified K-Fold CV
cv = StratifiedKFold(n_splits=3)
random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dist, n_iter=10, cv=cv, verbose=2, random_state=42, n_jobs=-1)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Print best parameters
print("Best Parameters:", random_search.best_params_)

# Predict on the testing data
y_pred = random_search.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print(classification_report(y_test, y_pred))

from sklearn.metrics import precision_recall_fscore_support

# Calculate precision, recall, and F1-score
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

# Print precision, recall, and F1-score for each class
for i in range(len(precision)):
    print(f"Class {i}:")
    print(f"  Precision: {precision[i]:.2f}")
    print(f"  Recall: {recall[i]:.2f}")
    print(f"  F1-score: {f1_score[i]:.2f}")

# Save the best model
dump(random_search.best_estimator_, 'insurance_claim_mlp_model.joblib')