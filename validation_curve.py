import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve


def plot_validation_curve(train_scores, test_scores, param_range, xscale):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')

    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color='blue')

    plt.plot(param_range, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(param_range,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xscale(xscale)
    plt.legend(loc='lower right')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.ylim([0.6, 1.0])
    plt.tight_layout()
    # plt.savefig('./figures/validation_curve.png', dpi=300)
    plt.show()


def plot_validation_curve_lr(X_train, y_train, est):
    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    train_scores, test_scores = validation_curve(
        estimator=est,
        X=X_train,
        y=y_train,
        param_name='clf__C',
        param_range=param_range,
        cv=10)

    plot_validation_curve(train_scores=train_scores, test_scores=test_scores, param_range=param_range, xscale='log')


def plot_validation_curve_knn(X_train, y_train, est):
    param_range = [1, 5, 10, 15, 20, 25, 30, 40, 50]

    train_scores, test_scores = validation_curve(
        estimator=est,
        X=X_train,
        y=y_train,
        param_name='clf__n_neighbors',
        param_range=param_range,
        cv=10)

    plot_validation_curve(train_scores=train_scores, test_scores=test_scores, param_range=param_range, xscale='linear')

def plot_validation_curve_dt(X_train, y_train, est):
    param_range = [1, 2, 3, 4, 5, 6, 7, 8]

    train_scores, test_scores = validation_curve(
        estimator=est,
        X=X_train,
        y=y_train,
        param_name='max_depth',
        param_range=param_range,
        cv=10)

    plot_validation_curve(train_scores=train_scores, test_scores=test_scores, param_range=param_range, xscale='linear')
