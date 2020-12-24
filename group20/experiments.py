from .model import model as md
from .model import validator as vd
from .timeml import timemldoc as td
from group20 import plots as pt
from nltk.metrics import (f_measure, accuracy)
from sklearn.metrics import (accuracy_score, f1_score)


def f1_data(filenames, states, tag_method, folds=[5,10], ngram=2):
    evt = vd.EventTaggerValidator(filenames, states, tag_method)
    for k in folds:
        mean, dev = evt.cross_validate(k)
        print(mean, dev)


def confusion_and_roc(filenames, states, tag_method):
    evt = vd.EventTaggerValidator(filenames, states, tag_method)
    evt.confusion_and_roc()


def plot_smooth():
    pt.plot_smooths()
    
def evaluate_hmm(filenames, states, tag_method, title, folds=[5]):
    """
    Does cross validation for the model with k = 5 
    and plots the mean and err of accuracy and f1
    """
    evt1 = vd.EventTaggerValidator(filenames, states, tag_method)
    train_mean_std = [[], []]
    test_mean_std = [[], []]
    print("accuracy")
    for k in folds:
        train, test = evt1.cross_validate(k, accuracy_score)
        print('train mean & std', train)
        print('test mean & std ', test)
        train_mean_std[0].append(train[0])
        train_mean_std[1].append(train[1])
        test_mean_std[0].append(test[0])
        test_mean_std[1].append(test[1])
        


    print("f1")
    
    train_mean_std = [[], []]
    test_mean_std = [[], []]
    for k in folds:
        train, test = evt1.cross_validate(k, f1_score, exclude=True)
        print('train mean & std', train)
        print('test mean & std ', test)
        train_mean_std[0].append(train[0])
        train_mean_std[1].append(train[1])
        test_mean_std[0].append(test[0])
        test_mean_std[1].append(test[1])
        


def baseline(filenames, states, tag_method, folds=[5]):
    print("baseline...")
    evt1 = vd.EventTaggerValidator(filenames, states, tag_method)
    evt1.baseline()


def plot_baseline_compare():
    pt.plot_baseline_model_comparison()
    
    








