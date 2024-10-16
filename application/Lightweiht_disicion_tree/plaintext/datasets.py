import numpy as np
from ucimlrepo import fetch_ucirepo

'''

Heart Disease
Accuracy of the decision tree model: 0.9133858267716536
max-depth: 5
Our Accuracy: 0.9304461942257218
'''
'''
Census Income
Accuracy of the decision tree model: 0.55
max-depth: 5
Our Accuracy: 0.58
'''

'''
Accuracy of the decision tree model: 0.9133858267716536
max-depth: 5
Our Accuracy: 0.9304461942257218
'''
rice_cammeo_and_osmancik = fetch_ucirepo(id=545)

# data (as pandas dataframes)
X = rice_cammeo_and_osmancik.data.features
y = rice_cammeo_and_osmancik.data.targets

# Numerical data imputation
X = X.dropna()  # Removes rows with any missing values in X
y = y.loc[X.index]  # Ensure y matches the cleaned X rows

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the decision tree model: {accuracy:.2f}")



'''
Accuracy of the decision tree model: : 0.82
Our Accuracy: 0.8555555555555555
max-depth: 6
'''

# fetch dataset
raisin = fetch_ucirepo(id=850)

# data (as pandas dataframes)
X = raisin.data.features
y = raisin.data.targets

# Numerical data imputation
X = X.dropna()  # Removes rows with any missing values in X
y = y.loc[X.index]  # Ensure y matches the cleaned X rows
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the decision tree model: {accuracy:.2f}")


'''
Accuracy of the decision tree model: 0.89
max-depth: 6
'''

# fetch dataset
wine = fetch_ucirepo(id=109)

# data (as pandas dataframes)
X = wine.data.features
y = wine.data.targets
# Numerical data imputation
X = X.dropna()  # Removes rows with any missing values in X
y = y.loc[X.index]  # Ensure y matches the cleaned X rows
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the decision tree model: {accuracy:.2f}")

"""
Accuracy of the decision tree model: 0.89
Our Accuracy: 0.9075144508670521
"""
from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
le = LabelEncoder()




# fetch dataset
car_evaluation = fetch_ucirepo(id=19)

# data (as pandas dataframes)
X = car_evaluation.data.features
y = car_evaluation.data.targets
X = X.dropna()  # Removes rows with any missing values in X
y = y.loc[X.index]  # Ensure y matches the cleaned X rows
# Assuming X is a DataFrame with all columns being categorical
X_encoded = X.apply(le.fit_transform)  # Apply label encoding to each column
# If y is also categorical
y_encoded = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2)
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the decision tree model: {accuracy:.2f}")


'''
Accuracy of the decision tree model: 0.93

'''
# fetch dataset
phishing_websites = fetch_ucirepo(id=327)

# data (as pandas dataframes)
X = phishing_websites.data.features
y = phishing_websites.data.targets
print(np.unique(y))
X = X.dropna()  # Removes rows with any missing values in X
y = y.loc[X.index]  # Ensure y matches the cleaned X rows
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the decision tree model: {accuracy:.2f}")

# fetch dataset
breast_cancer_wisconsin_original = fetch_ucirepo(id=15)
# data (as pandas dataframes)
X = breast_cancer_wisconsin_original.data.features
y = breast_cancer_wisconsin_original.data.targets
print(np.unique(y))
X = X.dropna()  # Removes rows with any missing values in X
y = y.loc[X.index]  # Ensure y matches the cleaned X rows
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the decision tree model: {accuracy:.2f}")


# '''
# Accuracy of the decision tree model: 70
#
# '''
# # fetch dataset
# covertype = fetch_ucirepo(id=31)
#
# # data (as pandas dataframes)
# X = covertype.data.features
# y = covertype.data.targets
#
# print(np.unique(y))
# X = X.dropna()  # Removes rows with any missing values in X
# y = y.loc[X.index]  # Ensure y matches the cleaned X rows
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# clf = DecisionTreeClassifier(max_depth=5)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy of the decision tree model: {accuracy:.2f}")