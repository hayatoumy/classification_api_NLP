import requests
import json
import pandas as pd
from time import sleep

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier


## Testing out the links:
#-------------------------
cook = 'https://api.pushshift.io/reddit/search/submission/?subreddit=cooking&size=100'
nut = 'https://api.pushshift.io/reddit/search/submission/?subreddit=nutrition&size=100'

res_cook = requests.get(cook)
sleep(1)
res_nut = requests.get(nut)

# making sure both urls are fine
print(res_cook.status_code)
print(res_nut.status_code)

# getting the dictionaries
cook_dict = res_cook.json()
nut_dict = res_nut.json()

# examining the dictionaries:
print(cook_dict.keys())
print(nut_dict.keys())

## Exploring the dictionaries: 
cook_dict['data'][0]
cook_dict['data'][0].keys()
len(cook_dict['data'])

cook_dict['data'][5]['selftext']

## Features and target variable:
cook_data = []
cook_target = []
for i,_ in enumerate(cook_dict['data']):
    if cook_dict['data'][i]['selftext'] !='':
        cook_data.append(cook_dict['data'][i]['selftext'])
        cook_target.append(cook_dict['data'][i]['subreddit'])
cook = pd.DataFrame(zip(cook_data, cook_target), columns = ['post', 'topic'])

nut_data = []
nut_target = []
for i,_ in enumerate(nut_dict['data']):
    if nut_dict['data'][i]['selftext']!='':
        nut_data.append(nut_dict['data'][i]['selftext'])
        nut_target.append(nut_dict['data'][i]['subreddit'])
nut = pd.DataFrame(zip(nut_data, nut_target), columns = ['post','topic'])

#merging the two dataframes together: 
df = pd.concat([cook, nut], axis = 0, sort = False)
df.shape

# more cleaning, dropping rows with '[removed]' posts 
removed_posts_indices = df.loc[df['post']=='[removed]', :].index
df.drop(removed_posts_indices, inplace = True)

### Collecting a 100 posts from 60 days ago till now, in reverse order: 
#------------------------------------------------------------------------
base_url_cook = 'https://api.pushshift.io/reddit/search/submission/?subreddit=cooking&size=100&before={}d'
urls_cook = [base_url_cook.format(i) for i in range(60,-1,-1)] # generate the urls
# the first -1 is the stopping point, coz range is exclusive to the endpoint. 
# the second -1 is to go in reverse on the range.
base_url_nut = 'https://api.pushshift.io/reddit/search/submission/?subreddit=nutrition&size=100&before={}d'
urls_nut = [base_url_nut.format(i) for i in range(60,-1,-1)]

pages_cook = []
for u in urls_cook:
    sleep(1)
    pages_cook.append(requests.get(u).json()['data'])

pages_nut = []
for u in urls_nut:
    sleep(1)
    pages_nut.append(requests.get(u).json()['data'])

cook_data = []
nut_data = []
cook_target = []
nut_target = []

for p in pages_cook: 
    for post in p:
        if post['selftext']!='':
            cook_data.append(post['selftext'])
            cook_target.append(post['subreddit'])

for p in pages_nut:
    count2 = 0
    for post in p:
        try: #because one post doesn't have a 'selftext'. nut_data stops at 116. therefore I need try/except
            if post['selftext']!='':
                nut_data.append(post['selftext'])
                nut_target.append(post['subreddit'])
        except: 
            nut_data.append('[removed]') # i want to add what I want to drop later on. some posts have '[removed]' in them
            nut_target.append('nutrition')
        
# making the each topic dataframe, then combining them into one dataframe
cook = pd.DataFrame(zip(cook_data, cook_target), columns = ['post', 'topic'])
cook.loc[cook['post']=='[removed]',:].index
cook.drop(cook.loc[cook['post']=='[removed]',:].index, inplace = True)

nut = pd.DataFrame(zip(nut_data, nut_target), columns = ['post', 'topic'])
nut.loc[nut['post']=='[removed]',:].index
nut.drop(nut.loc[nut['post']=='[removed]',:].index, inplace = True)

print(f'shape of the cleaned nutrition dataframe {nut.shape}')
print(f'shape of the cleaned cooking dataframe {cook.shape}')

#combining the dataframes
df = pd.concat([cook, nut], axis = 0, sort = False)
# saving it to a csv file
df.to_csv('reddit_cook_nut.csv') # don't forget to save, so you don't lose your work, when I run a for loop below

df = pd.read_csv('reddit_cook_nut.csv') # so I don't have to request every time.
print(f'shape of the combined and ready dataframe {df.shape}')
#---------------------------------------------------------------------
### EDA 
df['topic'].value_counts() # nutrition class will be considered 1, and cooking 0

# plotting most repeated words, excluding non-relevant
import matplotlib.pyplot as plt

more_stop = ['just', 'com', 'https', 'know', 'want', 'www', 've','x200b', 'really', 'like'] 
custom_words = list(ENGLISH_STOP_WORDS) + more_stop

cv = CountVectorizer(stop_words = custom_words) # ngram_range = (1,2) means i want both 1 and 2 words columns
sparse_mat = cv.fit_transform(df['post']) #fitting the model

all_feature_df = pd.DataFrame(sparse_mat.todense(), columns=cv.get_feature_names()) #attaching column names ie. words

all_feature_df.sum().sort_values(ascending=False).head(15).plot(kind='barh', figsize = (8,8));
plt.show();
# remember these are the lammatized words..
common_words_indicies = all_feature_df.sum().sort_values(ascending=False).head(20).index
# we need the indices because that's what .loc is looking for (or a boolean mask) when I use it below


## Most Frequent Word By Topic:
# making a sparse matrix of the words
cv = CountVectorizer(stop_words = custom_words)
X_train_cv = cv.fit_transform(X_train) # this is the sparse matrix
# making this sparse matrix into a dataframe
sparse = pd.DataFrame(X_train_cv.todense(), columns=cv.get_feature_names())

df_nutrition = sparse.loc[df['topic'] == 'nutrition'].copy()
df_cooking = sparse.loc[df['topic'] == 'Cooking'].copy()
# Making separate dataframe again, (assuming we're reading from the saved csv rather than requesing..)
#we use .copy() to create an actual copy of the subdata frame, otherwise whatever I do to the 
# subdata frame will be done to the original dataframe. b/c without .copy() the new subdata frames are
# just pointers to the original dataframe!

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,8), sharex=True) #last argument to set same scale for x

df_nutrition.sum().loc[common_words_indicies].plot(kind = 'barh',
                                          ax = ax[0], title = 'common words in relation to Nutrition posts'.title());
df_cooking.sum().loc[common_words_indicies].plot(kind = 'barh', ax= ax[1], 
                                        title = 'common words in relation to cooking posts'.title());

#------------------------------------------------------------------
### Modeling Part
# Train/test/split
X_train, X_test, y_train, y_test  = train_test_split(df['post'], df['topic'], stratify = df['topic'], random_state = 42)

# Creating Pipes & GridSearching on each & fitting & finding best score:
cv_log_pipe = Pipeline([('vector', CountVectorizer()), ('logreg', LogisticRegression())])
cv_log_params = {
    'vector__min_df' : [3,4,8],
    'vector__stop_words' : [custom_words, None],
    'logreg__penalty' : ['l1', 'l2'],
    'logreg__C' : [.1, 5, 50]
}
cv_log_gs = GridSearchCV(cv_log_pipe, cv_log_params, cv = 5, verbose=0)
cv_log_gs.fit(X_train, y_train)
print('best score of CountVectorizer with Logistic Regression is:', cv_log_gs.best_score_.round(4))
#------
cv_nb_pipe = Pipeline([('vector', CountVectorizer()) , ('naive', MultinomialNB())])
cv_nb_params = {
    'vector__min_df' : [3,4,8],
    'vector__stop_words' : [custom_words, None],
    'naive__alpha' : [0.001, 0.01, 0.1, 0.5, 0.8]
}
cv_nb_gs = GridSearchCV(cv_nb_pipe, cv_nb_params, cv = 5, verbose=0)
cv_nb_gs.fit(X_train, y_train)
print('best score of CountVectorizer with naive multinomial Bayes is:', cv_nb_gs.best_score_.round(4))
#-------
tf_log_pipe = Pipeline([('tfdf', TfidfVectorizer()),('logreg', LogisticRegression())])
tf_log_params = {
    'tfdf__stop_words' : [custom_words, None],
    'logreg__penalty' : ['l1', 'l2'],
    'logreg__C' : [.1, 5, 50]
}
tf_log_gs = GridSearchCV(tf_log_pipe, tf_log_params, cv = 5, verbose=0)
tf_log_gs.fit(X_train, y_train)
print('best score of TFDIF with Logistic Regression is:', tf_log_gs.best_score_.round(4))
#-------
tf_nb_pipe = Pipeline([('tfdf', TfidfVectorizer()),('naive', MultinomialNB())])
tf_nb_params = {
    'tfdf__stop_words' : [custom_words, None],
    'naive__alpha' : [0.001, 0.01, 0.1, 0.5, 0.8]
}
tf_nb_gs = GridSearchCV(tf_nb_pipe, tf_nb_params, cv = 5, verbose=0)
tf_nb_gs.fit(X_train, y_train)
print('best score of TFDIF with naive multinomial Bayes is:', tf_nb_gs.best_score_.round(4))
#---------
## Random Forest Model
X_train_cv = CountVectorizer().fit_transform(X_train)

forest = RandomForestClassifier()
forest_params = {
    'n_estimators' : [5, 10, 15, 20],
    'max_depth' : [10, 20, 30],
    'max_features' : [.2, .3]
}

# GridSearch over the random forest model
gs_forest = GridSearchCV(forest, param_grid = forest_params, cv = 5)
gs_forest.fit(X_train_cv, y_train)

# printing best scores of each model for comparison:
print('best score of random foresets is: ',gs_forest.best_score_.round(4))
print('best score of CountVectorizer with Logistic Regression is:', cv_log_gs.best_score_.round(4))
print('best score of CountVectorizer with naive multinomial Bayes is:', cv_nb_gs.best_score_.round(4))
print('best score of TFDIF with Logistic Regression is:', tf_log_gs.best_score_.round(4))
print('best score of TFDIF with naive multinomial Bayes is:', tf_nb_gs.best_score_.round(4))

#--------------------------------------------------------------------------
## the winner is TFDIF with Logistic Regression, so we're gonna fit that one
# fininding best estimators of the winning model to use them: 
print(tf_log_gs.best_estimator_.get_params()['steps']) 

# fitting the winning model and vectorizer:
tf = TfidfVectorizer(stop_words = custom_words)

X_train_tf = tf.fit_transform(X_train)
X_test_tf = tf.transform(X_test)
print(X_train_tf.shape)

logreg = LogisticRegression(C = 50, penalty = 'l2') #l2 is Ridge (google: 'l1 ridge or lasso?')
logreg.fit(X_train_tf, y_train)
preds = logreg.predict(X_test_tf)

print('accuracy score of TFDIF and Logistic on testing data: ', logreg.score(X_test_tf, y_test).round(4))
print('cross-validation score is: ', \
    cross_val_score(logreg, X_train_tf, y_train, cv = 5).mean().round(4)) #this is how we expect the model to perform on unseen data

# extra making sure it's not overfitting, by scoring it on train and test dataset
print('on training set, score is: ', tf_log_gs.score(X_train, y_train).round(4))
print('on training set, score is: ', tf_log_gs.score(X_test, y_test).round(4))

#-----------------------------------------------------------
### getting confusion matrix & classification report

print('classification report', classification_report(y_test, preds))

# to make confusion matrix look nicer: 
col_names = ['Predicted ' + i for i in df['topic'].value_counts().index]
index_names = ['Actual ' + i for i in df['topic'].value_counts().index]
cm = pd.DataFrame(confusion_matrix(y_test, preds), columns = col_names, index = index_names )
cm

### getting coefficients of the winning model and plotting them:
tf_log_gs.best_estimator_.steps[1][1].coef_ # same as logreg.coef_ 

# making coeficients into a dataframe to plot them
coef_df = pd.DataFrame(logreg.coef_, columns = tf.get_feature_names()).T.sort_values(by = 0).head(15)
coef_df['abs'] = coef_df[0].abs()
coef_df.sort_values(by = 'abs', ascending = False).head(15)

# plotting most important coefficients: 
coef_df.sort_values(by = 'abs', ascending = False).head(15)[0].plot(kind = 'barh', figsize = (10,10), title = "most \
important 15 coefficeints in our model, with respect to class: nutrition".title(), fontsize = 15);


#----------------------------------------------------
## Extra Exploration, Tweaking, and Evaluation
# getting prediction probabilities from our winning model, only for the positive class 
probs_nut = tf_log_gs.predict_proba(X_test)[:,1]


# setting the threshold, and getting the new predictions/classifications: 
def classify(thresh, probs_list):
    """
    thresh: threshold of classification 
    probs_list: a list of predict_proba of only the class of interest. Must be worked outside the function 
    """
    preds_thresh = ['nutrition' if probs_list[i] >= thresh else 'Cooking' for i in range(len(probs_list))]
    return preds_thresh


# getting the confusion matrix with the new threshold predictions:  
pd.DataFrame(confusion_matrix(y_test, classify(0.9, probs_nut)), columns = col_names, index = index_names )
# col_names, and index_names are defined above


# seeing the new classification report to check for precision: 
print(classification_report(y_test, classify(0.9, probs_nut)))


# Finally, visualizing our model performance, using ROC curve:
from sklearn.metrics import roc_curve

# to set up for ROC curve 
y_numerical = y_test.map({'nutrition' : 1 , 'Cooking' : 0})
fpr, tpr, _ = roc_curve(y_numerical, probs_nut)

plt.figure(figsize = (6,6))
plt.plot(fpr, tpr);
plt.plot([0,max(y_numerical)],[0, max(y_numerical)], '--'); # it takes only encoded numerical y
plt.title('ROC curve for the TFDIF vectorizer with logistic regression'.title());
plt.xlabel('false positive rate'.title());
plt.ylabel('true positive rate'.title());
#--------------------------------------------------------
### For Mere Visual Appeal; Creating a cover image of the most common words
import wordcloud

common_words = all_feature_df.sum().sort_values(ascending=False).head(20) # this is a series of words and their count
list(common_words.index)
common_words = ' '.join(list(common_words.index))

wc = wordcloud.WordCloud(background_color='gold', max_words=75, width=600, height =400)
# adding stop words will put only the stop words, not eliminate them.
cloud = wc.generate(common_words)
cloud.to_image()

#============================================================