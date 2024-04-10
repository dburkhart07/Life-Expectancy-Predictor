import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

# Load dataset
dataset = pd.read_csv('Religious_Practice_Survival_Data.csv')


# Separate the dataset into numerical and categorical features
numerical_features = dataset.select_dtypes(include=['int64', 'float64']).columns
categorical_features = dataset.select_dtypes(exclude=['int64', 'float64']).columns


# Use median imputation on the numerical features for missing values
dataset[numerical_features] = dataset[numerical_features].fillna(dataset[numerical_features].median())

# Use mode imputation on the categorical features for missing values
for col in categorical_features:
    dataset[col] = dataset[col].fillna(dataset[col].mode().iloc[0])


# One-hot encode categorical features
encoder = OneHotEncoder(drop='first', sparse=False)  # Drop first category to avoid dummy variable trap
encoded_categorical = encoder.fit_transform(dataset[categorical_features])

# Create DataFrame from encoded categorical features
encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))


# Combine numerical and encoded categorical features into X
X = pd.concat([dataset[numerical_features], encoded_categorical_df], axis=1)

# Define target variable y
y = dataset['died_2_year']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


'''
#Establish param-grid
param_dist = {
    'n_estimators': randint(50,100),
    'max_depth': randint(3,10),
    'min_samples_leaf': randint(1,10),
    'min_samples_split': randint(2,10)
}

random_search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42),
                                   param_distributions=param_dist,
                                   n_iter=5,  # Number of parameter settings that are sampled
                                   cv=5,
                                   random_state=42,
                                   scoring='accuracy',
                                   n_jobs=-1)


random_search.fit(X_train, y_train)


best_params = random_search.best_params_
print("Best parameters: ", best_params)
best_rfc = random_search.best_estimator_
'''

#Train the RandomForestClassifier based on best parameters
best_rfc = RandomForestClassifier(n_estimators=100, max_depth=9, min_samples_leaf=3, min_samples_split=8, random_state=42)
best_rfc.fit(X_train, y_train)


# Obtain accuracy
train_score = np.mean(cross_val_score(best_rfc, X_train, y_train, cv=5))
test_score = best_rfc.score(X_test, y_test)
print("Train Score:", train_score)
print("Test Score:", test_score)
