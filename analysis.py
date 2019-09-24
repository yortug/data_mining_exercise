# used to hold and transform data
import pandas as pd
import numpy as np

# used for data imbalance, upsampling
from sklearn.utils import resample
# used in scaling
from sklearn import preprocessing
# used in feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
# used for the actual training/test split for model evaluation
from sklearn.model_selection import train_test_split
# used for model fitting and cross validation
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
# used as metrics to compare the various classification models
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

# reading in data
df = pd.read_csv('data2019.student.csv')


# proportion of missing data table
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
# print(missing_value_df[missing_value_df['percent_missing'] > 0])


# drop ID because already have row indexes
# drop att13 and att19 due to their high amount of missing values
to_drop = ['ID', 'att13', 'att19']
df.drop(labels = to_drop, axis = 1, inplace = True)


# fill categorical missing values with mode of the column (att3 & att9)
df['att3'].fillna(value = df['att3'].mode()[0], inplace = True)
df['att9'].fillna(value = df['att9'].mode()[0], inplace = True)


# fill att25 missing values with median of column due to its large std.dev
df['att25'].fillna(value = int(df['att25'].median()), inplace = True)
# fill att28 missing values with mean of column due to its reasonable range
df['att28'].fillna(value = int(np.floor(df['att28'].mean())), inplace = True)


# proportion of missing data table AFTER cleaning (as expected)
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
# print(missing_value_df[missing_value_df['percent_missing'] > 0])


# find the values which are constant, unchanging, useless for analysis
# for i in df.columns:
#     if df[i].nunique() <= 1:
#         print(i)


# drop att14 and att17 as they've been found to be constant, unchanging throughout the column
to_drop = ['att14', 'att17']
df.drop(labels = to_drop, axis = 1, inplace = True)


# transpose dataframe, remove duplicate rows (now columns), then transpose again to put in to original form
# this gets rid of any duplicate columns; keeping the first instance of each duplicate set
df = df.T.drop_duplicates().T

# this gets rid of any duplicate rows, in which there is 300 within this dataset
df.drop_duplicates(inplace = True)

# INTERACTIVE PLOT: used in jupyter notebook to visualise data and assist in analysis (has dependencies)
# used to look at the boxplot generated for all columns, easy to spot numerical from nominal in this method
# @interact_manual
# def box_plots(attribute=list(df.columns)):
#     df[attribute].iplot(kind = 'box')


# attempt to cast all columns to numeric, ignore all errors (expected by string columns)
df = df.apply(pd.to_numeric, errors='ignore')


# create a manual list of the known numeric columns, easily found from the previous interactive visualisation
to_numeric = ['att18', 'att20', 'att21', 'att22', 'att25', 'att28']


# INTERACTIVE PLOT: used in jupyter notebook to visualise data and assist in analysis (has dependencies)
# used to see the column histograms for all numeric variables BEFORE scaling
# @interact_manual
# def hist_plots(attribute=list(to_numeric)):
#     df[attribute].iplot(kind = 'hist', 
#                         title = str(attribute) + ' Before Scaling',
#                        xTitle = 'value')


# split the numeric variables into a further two categories, found from the above interactive visualisation

# roughly normal looking variables
to_gauss = ['att18', 'att20', 'att21', 'att22']

# skewed, more log-normal, looking variables
to_ln = ['att25', 'att28']


# use the 'standard scaler' to deal with normally distributed columns
scaler = preprocessing.StandardScaler()
scaled = pd.DataFrame(scaler.fit_transform(df.loc[:,to_gauss].values), columns = to_gauss)
df.loc[:,to_gauss] = scaled.values

# use the 'robust scaler' to deal with skewed, log-normally distributed columns
scaler = preprocessing.RobustScaler() 
scaled = pd.DataFrame(scaler.fit_transform(df.loc[:,to_ln].values), columns = to_ln)
df.loc[:,to_ln] = scaled.values

# finally run the already scaled variables through min-max scaling to ensure they're within a [0,1] range
scaler = preprocessing.MinMaxScaler()
scaled = pd.DataFrame(scaler.fit_transform(df.loc[:,to_numeric].values), columns = to_numeric)
df.loc[:,to_numeric] = scaled.values


# INTERACTIVE PLOT: used in jupyter notebook to visualise data and assist in analysis (has dependencies)
# used to see the column histograms for all numeric variables AFTER scaling
# @interact_manual
# def hist_plots(attribute=list(to_numeric)):
#     df[attribute].iplot(kind = 'hist',
#                        title = str(attribute) + ' After Scaling',
#                        xTitle = 'value')


# manual list of all categorical columns, found from the initial interactive boxplot visualisation
to_categorical = ['att30', 'att29', 'att27', 'att26', 'att23', 'att16', 'att15', 'att12', 'att11', 
                  'att10', 'att9', 'att7', 'att6', 'att5', 'att4', 'att3', 'att2', 'att1']

# for each of these categorical columns, force each factor into its own column through a 'dummy expansion'
df = pd.get_dummies(df, columns = to_categorical)


# view the value count of the target class column, shows a significant imbalance
# df['Class'].value_counts()


# subset the dataframe to just the rows represented within the minority class (found to be == 0 here)
minority_class = df[df.Class == 0]

# resample the minority class, with replacement, of a size equal to the difference between majority & minority classes
minority_upsampled = resample(minority_class, replace = True, n_samples = (650 - 250))


# create an intermediate train & test datasets for the feature selection process (not indicative of final sets)

# everything but the last 100 rows (as they're set aside for final selection)
df_train = df[:-100]
# add the resampled set of rows to the end of the training set
df_train = pd.concat([df_train, minority_upsampled])
# reset the index (as indexes are carried through from resampling)
df_train = df_train.reset_index(drop = True)
# create y vector
y_train = df_train.loc[:,'Class'].values
# create X matrix
X_train = df_train.iloc[:,1:].values

# might as well create the true FINAL dataframe
df_test = df[-100:]


# create an intermediate X & y datasets for the feature selection process (not indicative of final sets)
Xx = df_train.iloc[:,1:]
yy = df_train.loc[:,'Class']


# selects the 10 best variables according to their chi^2 measure with the target variable
bestfeatures = SelectKBest(score_func = chi2, k = 10)
fit = bestfeatures.fit(Xx, yy)

# adds the variable names and their respective univariate importance scores to a dataframe
featureScores = pd.concat([pd.DataFrame(Xx.columns), pd.DataFrame(fit.scores_)], axis = 1)
featureScores.columns = ['Specs', 'Score']
# sorts the dataframe
featureScores = featureScores.set_index('Specs').sort_values(by = ['Score'], axis = 0, ascending = True)
# used for visualisation for the univariate variable importance
# featureScores.iplot(kind='barh', title = 'Univariate Feature Importance (K-Best from Chi^2)')

# takes the top 15 column names according to their univariate scoring
feat_list_1 = featureScores[-15:].index


# fits a standard ensemble tree method to assess the explained variance (feature importance)
model = ExtraTreesClassifier(n_estimators = 100)
model.fit(Xx,yy)

# look at the model's feature_importances attribute to assess all the variables
feat_importances = pd.Series(model.feature_importances_, index=Xx.columns)
# sort the values
feat_importances = feat_importances.sort_values()
# used for visualisation for the tree-based variable importance
# feat_importances.iplot(kind='barh', title = 'Tree-Based Feature Importance')

# takes the top 11 column names according to their tree based feature importance
feat_list_2 = feat_importances[-11:].index


# this part is used if we want to fit an svm to find important variables
# commented out as it was decided that logistic regression gave a better result
# lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(Xx, yy)
# model = SelectFromModel(lsvc, prefit=True)
# feat_list_3 = Xx.iloc[:,list(model.get_support(indices=True))].columns

# this part is used if we want to fit logistic regression to find important variables
lr = LogisticRegression(C = 0.000000001, penalty = 'l2', dual = False, solver = 'lbfgs').fit(Xx, yy)
model = SelectFromModel(lr, prefit=True)

# takes the found to be important column names according to importance in logistic regression
feat_list_3 = Xx.iloc[:,list(model.get_support(indices=True))].columns


# this is a list of all the common variables across all three variable seleciton methods
# these can likely be considered quite important as all three methods have picked them up
# list(set(feat_list_1) & set(feat_list_2) & set(feat_list_3))

# this 'master' list combines all the elements from the three variable selection methods
# overall it reduces our dimensionality from 79 down to 33, which definitely makes a more robust model (in most cases)
important_cols = list(set(list(feat_list_1) + list(feat_list_2) + list(feat_list_3)))


# as we have reduce the number of features taken through our training/testing
# we must also reduce our final testing 
X_test_final = df_test.loc[:,important_cols].values


# this is the main split; taking all the important columns from our
# cleaned, sclaed and balanced dataset, and assigning it to X
# along with taking just the target column and assigning it to y
X = df_train.loc[:,important_cols]
y = df_train.loc[:,'Class']


# the main train/test set split made easy using sklearn
# further splitting was done during the analysis to make a validation set from the training set here
# this code was conciously left out because parameter tuning has been done using the sklearn crossVal
# functions along with the sklearn pipelines, this code can be found roughly 100 lines of code down
# and an explanation is given in the model selection section of the report write-up
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42069)


# create an empty results dataframe for cross-validation results to be stored
results = pd.DataFrame()
# create an empty dataframe for the final proportions of ones/zeros from the last 100 rows
final_proportions = pd.DataFrame()


# create a dictonary containing all the classification models which are to be used
all_models = dict([
    ('kmeans', KMeans(n_clusters = 2)),
    ('rForest', RandomForestClassifier(n_estimators=100, max_depth=2)),
    ('logReg', LogisticRegression(solver = 'lbfgs', max_iter = 2500)),
    ('svm', svm.SVC(gamma='scale', decision_function_shape='ovo')),
    ('nBayes', MultinomialNB(alpha=2.5)),
    ('adaBoost', AdaBoostClassifier(n_estimators=100)),
    ('knearest', KNeighborsClassifier(n_neighbors=73, p=1, weights="distance"))
])

# iterate through all the models, running cross validation (folds = 10) using accuracy as a metric,
# then again using f1-score as a metric, append these as columns to the results df - ready to compare
# (also run predictions on the last 100 rows and calculate the proportion of zero/one predictions)
for i in all_models.keys():
    results[i + '_acc'] = cross_val_score(all_models[i], X_train, y_train, cv=10, scoring = 'accuracy')
    results[i + '_f1'] = cross_val_score(all_models[i], X_train, y_train, cv=10, scoring = 'f1')
    
    all_models[i].fit(X_train, y_train)
    pred_final = all_models[i].predict(X_test_final)
    zeros = np.bincount(pred_final.astype(int))[0]
    ones = np.bincount(pred_final.astype(int))[1]

    final_proportions[i] = [zeros, ones, (abs(ones - zeros) / 100), (1 - (abs(ones - zeros) / 100))]
             




# THIS CODE WAS USED REPEATEDLY TO TUNE THE MODEL PARAMETERS WHICH ARE SEEN ABOVE
# it was an overall manual process that I never fully automated because it was so tedious
# more information is given in the report writeup, within the classification selection section



# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV

# pipe = Pipeline([('classifier' , svm.SVC())])

# param_grid = [
# #     {'classifier' : [LogisticRegression()],
# #      'classifier__penalty' : ['l1', 'l2'],
# #     'classifier__C' : np.logspace(-4, 4, 20),
# #     'classifier__solver' : ['liblinear']},
# #     {'classifier' : [RandomForestClassifier()],
# #     'classifier__n_estimators' : list(range(10,101,10)),
# #     'classifier__max_features' : list(range(6,32,5))},
# #     {'classifier' : [MultinomialNB()],
# #     'classifier__alpha' : [0.2,0.5,0.6,1,1.5,2.5,5,10,20,50,75,90,130,250,500]},
#     {'classifier' : [svm.SVC()],
#     'classifier__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
#     'classifier__C' : [1,2,5,10,100,50,90],
#     'classifier__gamma' : ['scale', 1, 2, 3, 5, 0.5, 100, 25000],
#     'classifier__decision_function_shape' : ['ovr', 'ovo']}
    
# ]

# clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)


# best_clf = clf.fit(X_train, y_train)

# best_clf.best_estimator_
# best_clf.best_params_


# END OF MODEL PARAMETER TUNING CODE



# this code ensures that if this file is run through terminal or an IDE, the results dataframe will
# be printed to the screen for instant evaluation (along with the final proportions dataframe)
print()
print(results.to_string())

final_proportions.index = ['n_zeros', 'n_ones', 'difference', 'correct']
print()
print(final_proportions.to_string())


# creating an empty dataframe to store the final prediction results
df_predictions = pd.DataFrame()
# create the ID column manually, this ensures it's good (and is easiest)
df_predictions['ID'] = [i for i in range(1001, 1101)]

# k-nearest predictions get put into the 'Predict 1' column
all_models['knearest'].fit(X_train, y_train)
df_predictions['Predict 1'] = all_models['knearest'].predict(X_test_final)

# logistic regression predictions get put into the 'Predict 2' column
all_models['logReg'].fit(X_train, y_train)
df_predictions['Predict 2'] = all_models['logReg'].predict(X_test_final)

# this code has been commented out as it is the code used to output the final predictions to a csv
# this file is within the project directory so you have no need to run this (plus if you do this it could
# be detrimental as results may differ to models with random elements)
# df_predictions.to_csv('predict.csv', index = False)

# converting the final predictions to integer to ensure they are in the correct format as stated
# in the assignment sheet, and then printing it to the console/terminal/IDE
df_predictions = df_predictions.astype(int)

print()
print(df_predictions.to_string())



