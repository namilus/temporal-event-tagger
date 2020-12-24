from sklearn.model_selection import (KFold, train_test_split)
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, roc_curve)

from . import model as md
from ..timeml import timemldoc as td
import numpy as np
class EventTaggerValidator:
    """ Accuracy, cross val, plots, etc..."""
    def __init__(self, files, states, tag_method):
        self.states = states        
        self.files = np.array([[f] for f in files])
        self.X, self.y = EventTaggerValidator.create_X_y(files, tag_method)
        self.tag_method = tag_method


    def create_X_y(files, tagmethod):
        """ 
        Creates the typical X and y for ml
        X : np array of each sentence
        y : the labels i.e states
        """
        X = []
        y = []
        for f in files:
            d = td.TimeMLDoc(f)
            for s in d.sentences(tag=True, tag_method=tagmethod):
                words = [word for word, tag in s]
                tags = [tag for word, tag in s]
                X.append(words)
                y.append(tags)

        return np.array(X), np.array(y)

    def binary_labels(labels, pos_label):
        """
        Removes the start and end tags and changes to 0 and 1
        depending on the pos_label
        """
        new_labels = []
        for s_labels in labels:
            new_s_labels = list(s_labels[1:-1]) # remove 1st and last elements
            for i, label in enumerate(new_s_labels):
                if label == pos_label: new_s_labels[i] = 1
                else: new_s_labels[i] = 0

            new_labels.append(np.array(new_s_labels))
        return np.array(new_labels)
                
    
        
    def cross_validate(self, k, metric, exclude=False): # exlude is for f1 when there are no pos in the true labels
        kf = KFold(n_splits=k)
        metrics_train = []
        metrics_test = []
        for i, (train, test) in enumerate(kf.split(self.X)):
            print('fold', i)
            model = md.EventTaggerModel(self.states)
            model.fit(self.X[train], self.y[train])
            train_preds = EventTaggerValidator.binary_labels(model.predict(self.X[train]), 'E')
            test_preds = EventTaggerValidator.binary_labels(model.predict(self.X[test]), 'E')
            train_true = EventTaggerValidator.binary_labels(self.y[train], 'E')
            test_true = EventTaggerValidator.binary_labels(self.y[test], 'E')

            f_train_metrics = self.__metrics_per_sentence(train_true, train_preds, metric, exclude)
            f_test_metrics  = self.__metrics_per_sentence(test_true, test_preds, metric, exclude)

            print("fold train mean & std: ", f_train_metrics.mean(), f_train_metrics.std())
            print("fold test mean & std: ", f_test_metrics.mean(), f_test_metrics.std())
            metrics_train.append(f_train_metrics.mean())
            metrics_test.append(f_test_metrics.mean())


        metrics_train = np.array(metrics_train)
        metrics_test = np.array(metrics_test)

        return (metrics_train.mean(), metrics_train.std()), (metrics_test.mean(), metrics_test.std())
                                

    def __metrics_per_sentence(self, true, preds, metric, exclude):
        """ Returns array of metrics for each sentence prediction """
        metrics = []
        for t, p in zip(true, preds):
            if sum(t) > 0 or not exclude:
                metrics.append(metric(t, p))
        return np.array(metrics)


    def confusion_and_pr(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        model = md.EventTaggerModel(self.states)
        model.fit(X_train, y_train)
        true = EventTaggerValidator.binary_labels(y_test, 'E')
        preds = EventTaggerValidator.binary_labels(model.predict(X_test), 'E')
        ttn = 0; tfp = 0; tfn = 0; ttp = 0;
        for t, p in zip(true, preds):
            #print(confusion_matrix(t, p, labels=[0,1]).ravel())
            tn, fp, fn, tp = confusion_matrix(t, p, labels=[0,1]).ravel()
            ttn+=tn; tfp+=fp; tfn+=fn; ttp+= tp
        print("TN FP FN TP")
        print(ttn, tfp, tfn, ttp)



    def baseline(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
                
        model = md.Baseline(self.states)
        true = EventTaggerValidator.binary_labels(y_test, 'E')
        preds = EventTaggerValidator.binary_labels(model.predict(X_test), 'E')
        ttn = 0; tfp = 0; tfn = 0; ttp = 0;
        for t, p in zip(true, preds):
            tn, fp, fn, tp = confusion_matrix(t, p, labels=[0,1]).ravel()
            ttn+=tn; tfp+=fp; tfn+=fn; ttp+= tp

        acc_metrics = self.__metrics_per_sentence(true, preds, accuracy_score, False)
        f1_metrics = self.__metrics_per_sentence(true, preds, f1_score, True)
        print("acc", acc_metrics.mean(), acc_metrics.std())
        print("f1",  f1_metrics.mean(), f1_metrics.std())
        print("TN FP FN TP")
        print(ttn, tfp, tfn, ttp)


        
