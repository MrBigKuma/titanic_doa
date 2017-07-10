from sklearn.model_selection import GridSearchCV


def grid_search(clf_name, X_train, y_train, est, param_grid):
    # Setup GridSearch
    gs = GridSearchCV(estimator=est,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10,
                      n_jobs=-1)
    gs = gs.fit(X_train, y_train)

    print(clf_name, gs.best_score_)
    print(gs.best_params_)


def grid_search_svc(X_train, y_train, est):
    param_range = [0.1, 1.0, 10.0]
    param_grid = [{'clf__C': param_range,
                   'clf__kernel': ['linear']},
                  {'clf__C': param_range,
                   'clf__gamma': param_range,
                   'clf__kernel': ['rbf']}]

    grid_search('svc', X_train, y_train, est, param_grid)


def grid_search_lr(X_train, y_train, est):
    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'clf__C': param_range}]

    grid_search('lr', X_train, y_train, est, param_grid)


def grid_search_knn(X_train, y_train, est):
    param_range = [1, 5, 10, 15, 20, 25, 30, 50]
    param_grid = [{'clf__n_neighbors': param_range}]

    grid_search('knn', X_train, y_train, est, param_grid)


def grid_search_dt(X_train, y_train, est):
    param_range = [1, 2, 3, 4, 5, 6, 7, 8]
    param_grid = [{'max_depth': param_range}]

    grid_search('dt', X_train, y_train, est, param_grid)


def grid_search_rf(X_train, y_train, est):
    param_range = [80, 90, 100, 110, 120, 130, 140, 150]
    param_grid = [{'n_estimators': param_range}]

    grid_search('rf', X_train, y_train, est, param_grid)
