import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# from titanic_doa.grid_search import grid_search
from titanic_doa.grid_search import grid_search_knn, grid_search_rf, grid_search_lr, grid_search_dt

df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
# Label encoding
lbl = LabelEncoder()
lbl.fit(np.unique(list(df['Embarked'].values) + list(df_test['Embarked'].values)))
df['Embarked'] = lbl.transform(list(df['Embarked'].values))
df_test['Embarked'] = lbl.transform(list(df_test['Embarked'].values))
# df.drop(['Embarked'], axis=1, inplace=True)
# df_test.drop(['Embarked'], axis=1, inplace=True)

df['Family'] = df["Parch"] + df["SibSp"]
df['Family'].loc[df['Family'] > 0] = 1
df['Family'].loc[df['Family'] == 0] = 0

df_test['Family'] = df_test["Parch"] + df_test["SibSp"]
df_test['Family'].loc[df_test['Family'] > 0] = 1
df_test['Family'].loc[df_test['Family'] == 0] = 0

# drop Parch & SibSp
df = df.drop(['SibSp', 'Parch'], axis=1)
df_test = df_test.drop(['SibSp', 'Parch'], axis=1)

# df.loc[df['Sex'] == 'female', 'Sex'] = 0
# df.loc[df['Sex'] == 'male', 'Sex'] = 1

sex_label_encoder = {'male': 1, 'female': 0}
df['Sex'] = df['Sex'].map(sex_label_encoder)
df_test['Sex'] = df_test['Sex'].map(sex_label_encoder)

# X = df[picked_labels][:]
# y = df['Survived'][:]
# X_test = df_test[picked_labels][:]
# ---------- Pre-process Data

# Replace NaN with mean
# df_all = pd.concat([df, df_test])
df_all = df
imr_age = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr_age.fit(df_all['Age'].values.reshape(-1, 1))
df.loc[:, 'Age'] = imr_age.transform(df['Age'].values.reshape(-1, 1))
df_test.loc[:, 'Age'] = imr_age.transform(df_test['Age'].values.reshape(-1, 1))

imr_fare = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr_fare.fit(df_all['Fare'].values.reshape(-1, 1))
df_test.loc[:, 'Fare'] = imr_fare.transform(df_test['Fare'].values.reshape(-1, 1))

# Data visualization
# sns.set(style='whitegrid', context='notebook')
# sns.pairplot(df[['Survived', 'Pclass', 'Sex', 'Age', 'Family', 'Fare', 'Embarked']], size=2.5)
# plt.tight_layout()
# plt.show()

# Correlation heat map
# visualized_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'Family', 'Fare', 'Embarked']
# cm = np.corrcoef(df[visualized_cols].values.T)
# sns.set(font_scale=1.5)
# hm = sns.heatmap(cm,
#                  cbar=True,
#                  annot=True,
#                  square=True,
#                  fmt='.2f',
#                  annot_kws={'size': 15},
#                  yticklabels=visualized_cols,
#                  xticklabels=visualized_cols)

# plt.tight_layout()
# plt.savefig('./figures/corr_mat.png', dpi=300)
# plt.show()

picked_labels = ['Pclass', 'Sex', 'Age', 'Family', 'Fare', 'Embarked']
# X_train, X_test, y_train, y_test = train_test_split(df[picked_labels], df['Survived'], test_size=0.30, random_state=1)
X_train = df[picked_labels]
y_train = df['Survived']
X_test = df_test[picked_labels]

# scaler = StandardScaler()
# X_scale = scaler.fit_transform(X_train)
# pca = PCA(7)
# pca.fit_transform(X_scale, y_train)
# print("explained_variance_ratio_:")
# print(pca.explained_variance_ratio_)
# [ 0.3376731   0.21826022  0.12253677  0.10841108  0.08855146  0.07037803  0.05418933]

# ---------- Train

clf_svc = SVC(C=10.0, kernel='rbf', gamma=0.1, random_state=0, probability=True)
clf_lr = LogisticRegression(penalty='l2', C=0.1, random_state=0)
clf_knn = KNeighborsClassifier(n_neighbors=10, weights='uniform')
clf_dt = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=0)
clf_rf = RandomForestClassifier(n_estimators=200, random_state=0)

clf_dt_ada = AdaBoostClassifier(base_estimator=clf_dt,
                                n_estimators=500,
                                learning_rate=0.1,
                                random_state=0)

pipe_svc = Pipeline([
    ('scl', StandardScaler()),
    # ('pca', PCA(n_components=3)),
    ('clf', clf_svc)])

pipe_lr = Pipeline([
    ('scl', StandardScaler()),
    # ('pca', PCA(n_components=3)),
    ('clf', clf_lr)])

pipe_knn = Pipeline([
    ('scl', StandardScaler()),
    ('clf', clf_knn)])

clf_ensemble = VotingClassifier(estimators=[
    # ('svc', pipe_svc),
    ('lr', pipe_lr),
    ('knn', pipe_knn),
    ('dt', clf_dt),
    ('rf', clf_rf)],
    voting='soft')

# print(pipe_knn.get_params())
# print(clf_dt.get_params())

# grid_search_svc(X_train[:500], y_train[:500], pipe_svc)
grid_search_lr(X_train, y_train, pipe_lr)
grid_search_knn(X_train, y_train, pipe_knn)
grid_search_dt(X_train, y_train, clf_dt)
grid_search_rf(X_train, y_train, clf_rf)

# plot_validation_curve_lr(X_train, y_train, pipe_lr)
# plot_validation_curve_knn(X_train, y_train, pipe_knn)
# plot_validation_curve_dt(X_train, y_train, clf_dt)

# plot_learning_curve(X_train, y_train, pipe_svc)
# plot_learning_curve(X_train, y_train, pipe_lr)
# plot_learning_curve(X_train, y_train, pipe_knn)
# plot_learning_curve(X_train, y_train, clf_dt)
# plot_learning_curve(X_train, y_train, clf_dt_ada)
# plot_learning_curve(X_train, y_train, clf_rf)
# plot_learning_curve(X_train, y_train, clf_ensemble)

# ensemble learning
all_clf = [pipe_svc, pipe_lr, pipe_knn, clf_dt, clf_dt_ada, clf_rf, clf_ensemble]
clf_labels = [
    'svc',
    'lr',
    'knn',
    'dt',
    'dt_ada',
    'rf',
    'ensemble']

print('10-fold cross validation:\n')
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# plot_roc_auc(X_train, y_train, X_test, y_test, all_clf, clf_labels)

# scores = cross_val_score(estimator=clf_rf,
#                          X=X_train,
#                          y=y_train,
#                          cv=10,
#                          scoring='accuracy')
# print("ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), 'clf_ensemble'))

# ---------- Test
chosen_est = clf_ensemble
chosen_est.fit(X_train, y_train)
predictions = chosen_est.predict(X_test)
submission = pd.DataFrame({'PassengerId': df_test['PassengerId'],
                           'Survived': predictions})
submission.to_csv("submission.csv", index=False)
