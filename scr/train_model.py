import preprocessing as pr
import pickle
from sklearn import metrics
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('always')


class Class_Fit(object):

    def __init__(self, clf, params=None):
        if params:
            self.clf = clf(**params)
        else:
            self.clf = clf()

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)

    def grid_search(self, parameters, Kfold):
        self.grid = GridSearchCV(estimator=self.clf,
                                 param_grid=parameters,
                                 n_jobs=6,
                                 cv=Kfold)

    def grid_fit(self, X, Y):
        self.grid.fit(X, Y)


def print_result(Y, predictions):
    try:
        print("Precision: {:.2f} % ".format(
            100 * metrics.precision_score(Y, predictions)))
        print("Accuracy: {:.2f} % ".format(
            100 * metrics.accuracy_score(Y, predictions)))
        print("Recall: {:.2f} % ".format(
            100 * metrics.recall_score(Y, predictions, average='binary')))
        print("F1_score: {:.2f} % ".format(
            100 * metrics.f1_score(Y, predictions, average='micro')))
        print("AUC&ROC: {:.2f} % ".format(
            100 * metrics.roc_auc_score(Y, predictions)))
        count = list(predictions).count(False)
        print('The count of False is:', count)
        count_true = list(predictions).count(True)
        print('The count of True is:', count_true)
        print("____________________")
    except ValueError:
        pass


def fit_model(X_train, Y_train, X_2, Y_2, X_3, Y_3):
    """ Learn the classifier, prints metrics"""
    #Gradient Boosting Classifier
    gb = Class_Fit(clf=ensemble.GradientBoostingClassifier)
    param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
    gb.grid_search(parameters=param_grid, Kfold=5)
    gb.grid_fit(X=X_train, Y=Y_train)
    # объединяем
    gb_best = ensemble.GradientBoostingClassifier(**gb.grid.best_params_)
    votingC = ensemble.VotingClassifier(estimators=[('gb', gb_best)],
                                        voting='soft')
    # и обучаю его
    votingC = votingC.fit(X_train, Y_train)

    predictions_baseline = votingC.predict(X_2)  # baseline
    print("____________________")
    print('Balanced sampling baseline model metrics:')
    print_result(Y_2, predictions_baseline)
    predictions_3 = (votingC.predict_proba(X_2)[:, 1] >= 0.8).astype(bool)
    print('Balanced sampling threshold metrics :')
    print_result(Y_2, predictions_3)
    predictions_4 = (votingC.predict_proba(X_3)[:, 1] >= 0.8).astype(bool)
    print('Only prof threshold metrics:')
    print_result(Y_3, predictions_4)
    predictions_5 = votingC.predict(X_3)
    print('Only prof metrics with baseline classifier:')
    print_result(Y_3, predictions_5)
    return votingC


def dump_classifier():
    """dump our classifier to pickle"""
    X_train, Y_train, X_2, Y_2, X_3, Y_3 = pr.get_train_test_sets()

    votingC = fit_model(X_train, Y_train, X_2, Y_2, X_3, Y_3)
    with open('data/votingC.pkl', 'wb') as fid:
        pickle.dump(votingC, fid)
    return votingC


if __name__ == '__main__':
    dump_classifier()
