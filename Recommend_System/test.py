import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

books = pd.read_csv("./data/BX-Books.csv", sep=';', on_bad_lines='skip', encoding="latin-1", low_memory=False)
users = pd.read_csv("./data/BX-Users.csv", sep=';', on_bad_lines='skip', encoding="latin-1")
ratings = pd.read_csv("./data/BX-Book-Ratings.csv", sep=';', on_bad_lines='skip', encoding="latin-1")

print(f"Books: {len(books)}")
print(f"Users: {len(users)}")
print(f"Ratings: {len(ratings)}")

# Remove ratings value=0
ratings = ratings.loc[ratings['Book-Rating'] > 0]

# Remove books rated less than min_ratings
MIN_RATES = 10
counts = ratings['ISBN'].value_counts()
ratings = ratings[ratings['ISBN'].isin(counts[counts >= MIN_RATES].index)]

# ???????? if neccessary
books = books.dropna()
users = users.dropna()

print(f"Books: {len(books)}")
print(f"Users: {len(users)}")
print(f"Ratings: {len(ratings)}")

# Cascade
data = pd.merge(ratings, users, on='User-ID', how='inner')
data = pd.merge(data, books, on='ISBN', how='inner')

print(f"data: {len(data)}")

# Convert ratings to binary labels
y = data['Book-Rating'].apply(lambda x: 0 if x <= 6 else 1)

# Encode categorical features
from sklearn.preprocessing import LabelEncoder
features = ['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Location', 'Age']
X = data[features]
label_encoder = LabelEncoder()
for feature in features:
    X[feature] = label_encoder.fit_transform(X[feature])

print("Features after encoding:")
print(X.head())

# Data Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)

print("Train:", len(X_train))
print("Test:", len(X_test))


def get_model(module_name):
    if module_name == "BernoulliNB":
        model = BernoulliNB()
    elif module_name == "GaussianNB":
        model = GaussianNB()
    elif module_name == "MultinomialNB":
        model = MultinomialNB()
    elif module_name == "ComplementNB":
        model = ComplementNB()
    else:
        model = SVC(C=1, kernel='poly', degree=8, gamma='scale', coef0=0.0, 
              shrinking=True, probability=False, tol=1e-4, cache_size=200,
              class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovo', random_state=500)

    return model

# Initialize the list to store the results
results = []

module_list = ['BernoulliNB', 'GaussianNB', 'MultinomialNB', 'ComplementNB']

for module in module_list:
    model = get_model(module)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)
    results.append(report['macro avg'])

    # Print the classification report
    print(f"Using module: {module}")
    print(classification_report(y_test, predictions))

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'Confusion Matrix for module {module}')
    plt.show()


# # Train and evaluate the model for each parameter
# for param in parameters:
#     model = svm.SVC(kernel=param['kernel'], C=param['C'])
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#     report = classification_report(y_test, predictions, output_dict=True)
#     results.append(report['macro avg'])

#     # Print the classification report
#     print(f"Classification report for parameters {param}:")
#     print(classification_report(y_test, predictions))

#     # Plot the confusion matrix
#     cm = confusion_matrix(y_test, predictions)
#     sns.heatmap(cm, annot=True, fmt='d')
#     plt.title(f'Confusion Matrix for parameters {param}')
#     plt.show()