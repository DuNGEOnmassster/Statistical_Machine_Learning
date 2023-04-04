import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pickle
import seaborn as sns

def show_sns(data, target):
    cat = pd.concat((data,target),axis=1)
    cat.corr()
    plt.figure(figsize=(28,28))
    sns.heatmap(cat.corr(),cmap="RdBu_r", annot=True)


def test_svm(val_gen, model_path):
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    for iteration, batch in enumerate(val_gen):
        if iteration >= 1:
            break

        val_images, val_label = batch[0], batch[1]
        val_pred  = clf.predict(val_images)

        accuracy = accuracy_score(val_label, val_pred)
        report = classification_report(val_label, val_pred)
        print("Test Accuracy:", accuracy)
        print("Classification report:", report)


def fit_one_epoch(train_gen, val_gen):
    loss = 0
    train_set = set()
    need_cv = False
    need_test = True
    model_path = './checkpoints/svm_poly.pkl'

    print('Start Train Poly')

    # model = SVC(kernel='linear', C=1.0, gamma='scale')
    model =   SVC(C=1, kernel='poly', degree=8, gamma='scale', coef0=0.0, 
              shrinking=True, probability=False, tol=1e-4, cache_size=200,
              class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovo', random_state=500)

    parameters = {
        "C": [1],
        "kernel": ["poly"],
        "gamma": ["scale"],
        "decision_function_shape": ['ovo'],
        "tol": [1e-4],
        "degree": [7,8,9],
        "random_state": [500],
    }

    # rfe = RFE(estimator=SVC, n_features_to_select=14, step=1)


    for iteration, batch in enumerate(train_gen):
        # if iteration >= 1:
        #     break

        train_images, train_label = batch[0], batch[1]  # image (B,C,H,W)   label (B)
        # print(np.shape(train_images), np.shape(train_label))
        # model.fit(train_images, train_label)
        # print(iteration)
        if need_cv:
            gsearch = GridSearchCV(model, param_grid=parameters, scoring='accuracy', cv=10)
            gsearch.fit(train_images, train_label)
            print("Best score: %0.4f" % gsearch.best_score_)
            print("Best parameters set:")
            best_parameters = gsearch.best_estimator_.get_params()
            for param_name in sorted(parameters.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))
            model.fit(train_images, train_label)
        else:
            model.fit(train_images, train_label)
        train_pred = model.predict(train_images)
        accuracy = accuracy_score(train_label, train_pred)
        print(f"iteration {iteration}: train acc: {accuracy}")
        
        # save
    with open(model_path,'wb') as f:
        pickle.dump(model,f)

        # rfe.fit(train_images, train_label)
        # print(f"ranking = {rfe.ranking_}")
    
    if need_test:
        print('Start test')
        test_svm(val_gen, model_path)


    # acc = 0


