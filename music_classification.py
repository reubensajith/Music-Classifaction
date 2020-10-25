import matplotlib
import matplotlib.pyplot as plt

from sklearn.grid_search import GridSearchCV


from sklearn.model_selection import KFold # import KFold
from sklearn.metrics import classification_report

def load_train_data():
  '''Returns training dataset as dataframe, input features X and target classes y'''
  train_df = pd.read_csv('training_data.csv')
  X = train_df.iloc[:,0:13] 
  y = train_df['label'] 
  return train_df, X, y

def train_model_for_submission(train_df, model,scaled=False):
  '''Trains the model on whole training data and returns the trained model'''
  X = train_df.iloc[:,0:13] 
  y = train_df['label']
  if scaled:
  	X = normalise_data(X) 
  clf = model.fit(X, y)
  return clf

def predict_for_submission(model, scaled=False):
  '''Predicts the labels for data in submission file using the trained model as input and
  returns the predicted labels'''
  test_df = pd.read_csv('songs_to_classify.csv')
  if scaled:
  	test_df = normalise_data(test_df)
  y_pred = model.predict(test_df.to_numpy())
  return y_pred

def validate_model1(input_features, target, model, splits):
  '''Function which splits the data into n folds and validates the model performance'''
  kf = KFold(n_splits=splits, random_state=None, shuffle=True)
  for train_index, test_index in kf.split(input_features):
    X_train, X_test = input_features.iloc[train_index], input_features.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)    
    print(report)
    # print(pd.crosstab(np.array(y_pred), np.array(y_test)), '\n')
  print("=====================================================================")
  return report

def validate_model(input_features, target, model, splits):
  '''Function which splits the data into n folds and validates the model performance'''
  kf = KFold(n_splits=splits, random_state=None, shuffle=True)
  for train_index, test_index in kf.split(input_features):
    X_train, X_test = input_features.iloc[train_index], input_features.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    clf = model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)    
    print(report)
    print(pd.crosstab(np.array(y_pred), np.array(y_test)), '\n')
  print("=====================================================================")

from sklearn import preprocessing
def normalise_data(X):
  x = X.values #returns a numpy array
  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(x)
  X_scaled = pd.DataFrame(x_scaled)
  return X_scaled

def evaluateKNN(train_df, y):
	iterations = 50
	splits = 3
	results = []

	for k in range(iterations):
		clf = KNeighborsClassifier(n_neighbors=k+1)
		kf = KFold(n_splits=splits, random_state=None, shuffle=True)
		split_list=[]
		for train_index, test_index in kf.split(train_df):
			X_train, X_test = train_df.iloc[train_index], train_df.iloc[test_index]
			y_train, y_test = y.iloc[train_index], y.iloc[test_index]
			clf = model.fit(X_train, y_train)
			y_pred = clf.predict(X_test)
			split_list.append(np.mean(y_pred != y_test))
		results.append(split_list)

	# Plotting the misclassification rate as a scatter plot for 3 folds 
	import matplotlib.pyplot as plt

	mis1 = []
	mis2 = []
	mis3 = []
	for i in range(50):
	  mis1.append(results[i][0])
	  mis2.append(results[i][1])
	  mis3.append(results[i][2])
		  
	# for fold 1  
	K = np.linspace(1,50,50)
	plt.plot(K, mis1, '.')
	plt.ylabel("Misclassification")
	plt.xlabel("Number of neighbors")
	plt.show()

	# for fold 2
	K = np.linspace(1,50,50)
	plt.plot(K, mis2, '.')
	plt.ylabel("Misclassification")
	plt.xlabel("Number of neighbors")
	plt.show()

	# for fold 3

	K = np.linspace(1,50,50)
	plt.plot(K, mis3, '.')
	plt.ylabel("Misclassification")
	plt.xlabel("Number of neighbors")
	plt.show()


# KNN on raw data
evaluateKNN(X, y)

# KNN for normalised data
evaluateKNN(X_scaled, y)
	
	
model = LogisticRegression(solver='lbfgs',penalty='l2', class_weight='balanced')
print(model)
validate_model(X,y, model, 3)

# logistic regression model
model = LogisticRegression(solver='liblinear', penalty='l1', class_weight='balanced')
model = train_model_for_submission(train_df, model)
predicted_values = predict_for_submission(model)

# Scatter plot for the label vs speechiess
train_df.plot.scatter(x='label',y='speechiness');

# Scatter plot for the label vs danceability
train_df.plot.scatter(x='label',y='danceability');

# Scatter plot for the label vs acousticness
train_df.plot.scatter(x='label',y='acousticness');

# Scatter plot for the label vs speechiness vs danceability vs acousticness
sns.set()
cols = ['label','speechiness','danceability','acousticness']
sns.pairplot(train_df[cols], height = 5)
plt.show();


#Random Forest classifier model for the raw data
train_df = pd.read_csv('training_data.csv')
X = train_df.iloc[:,0:13] 
y = train_df['label'] 

model= RandomForestClassifier(n_estimators=500,min_samples_split=25)
print(model)
validate_model(X,y, model, 3)

# Random Forest classifier model for the normalized data
X_scaled = normalise_data(X)

model= RandomForestClassifier(n_estimators=500,min_samples_split=25)
print(model)
validate_model(X_scaled,y, model, 3)

# Random Forest classifier model for the raw data with new features
model= RandomForestClassifier(n_estimators=500,min_samples_split=20)
print(model)
validate_model(X_new,y, model, 3)

# Random Forest classifier model for the normalized data with the new features
X_scaled_new= normalise_data(X_new)

model= RandomForestClassifier(n_estimators=500,min_samples_split=20)
print(model)
validate_model(X_scaled_new,y, model, 3)

#Pandas profiling to find the weightage of the attributes
from pandas_profiling import ProfileReport
profile = ProfileReport(df, title='Pandas Profiling Report', html={'style':{'full_width':True}})

# Decision Tree Classifier for raw data
model= tree.DecisionTreeClassifier(max_depth=2)
print(model)
validate_model(X_new,y, model, 3)

#Decision Tree Classifier for normalised data
X_scaled=normalise_data(X)

model= tree.DecisionTreeClassifier(max_depth=2)
print(model)
validate_model(X_scaled,y, model, 3)


# LDA model on raw data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X = train_df.iloc[:,0:13] 
y = train_df['label'] 


model = LinearDiscriminantAnalysis()
print(model)
validate_model(X,y, model, 3)

# LDA for normalised data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X = train_df.iloc[:,0:13] 
y = train_df['label'] 


X_scaled = normalise_data(X)

model = LinearDiscriminantAnalysis()
print(model)
validate_model(X_scaled,y, model, 3)

# QDA for raw data
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

X = train_df.iloc[:,0:13] 
y = train_df['label'] 


model = QuadraticDiscriminantAnalysis()
print(model)
validate_model(X,y, model, 3)

# QDA for normalised data
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

X = train_df.iloc[:,0:13] 
y = train_df['label'] 


X_scaled = normalise_data(X)

model = QuadraticDiscriminantAnalysis()
print(model)
validate_model(X_scaled,y, model, 3)

