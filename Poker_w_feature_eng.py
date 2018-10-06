# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
import pickle

from pandas.plotting import scatter_matrix
plt.style.use('ggplot')

from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import svm
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# df1 will house the raw data.
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data"
names = ["S1","C1","S2","C2","S3","C3","S4","C4","S5","C5","Hand"] 
df1 = pd.read_csv(url,names=names)   


# separate out the suits, from the cards.

cards = df1[['C1','C2','C3','C4','C5']]
suits = df1[['S1','S2','S3','S4','S5']]

# create new features dataframe
columns = ['1 Pair', '2 Pair', '3 of a kind', '4 of a kind', 'flush' ]
features = pd.DataFrame(index = range(0,20510),columns = columns)
    

# for loop: performs count action on each row.
# if there is a 2 in the counter.values, write 1 to corresponding feature dataframe cell.

for i in range(0, 25010):
    row = cards.iloc[i,:]
    row = np.array(row)
    counter = collections.Counter(row)
    freq = list(counter.values())
    if 2 in freq:
        features.loc[i,'1 Pair'] = 1
    else:
        features.loc[i,'1 Pair'] = 0
        
# process is repeated for 2 pair:

for i in range(0, 25010):
    row = cards.iloc[i,:]
    row = np.array(row)
    counter = collections.Counter(row)
    freq = list(counter.values())
    counter_2 = collections.Counter(freq)
    freq_2 = list(counter_2.values())
    if 2 in freq_2:
        features.loc[i,'2 Pair'] = 1
    else:
        features.loc[i,'2 Pair'] = 0
        

# process is repeated for 3 of a kind:

for i in range(0, 25010):
    row = cards.iloc[i,:]
    row = np.array(row)
    counter = collections.Counter(row)
    freq = list(counter.values())
    if 3 in freq:
        features.loc[i,'3 of a kind'] = 1
    else:
        features.loc[i,'3 of a kind'] = 0

# repeat for 4 of a kind

for i in range(0,25010):
    row = cards.iloc[i,:]
    row = np.array(row)
    counter = collections.Counter(row)
    freq = list(counter.values())
    if 4 in freq:
        features.loc[i,'4 of a kind'] = 1
    else:
        features.loc[i,'4 of a kind'] = 0
        
#repeat for flush:

for i in range(0,25010):
    row = suits.iloc[i,:]
    row = np.array(row)
    counter = collections.Counter(row)
    freq = list(counter.values())
    if 5 in freq:
        features.loc[i,'flush'] = 1
    else:
        features.loc[i,'flush'] = 0

#combine features and original data set.

df_X = pd.concat([features,df1],axis =1)
X = np.array(df_X.drop(['Hand'],1))
y = np.array(df_X['Hand'])


X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size = 0.2)


# scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# MLP Classifier

mlp = MLPClassifier(hidden_layer_sizes = (52,26))
mlp.fit(X_train,y_train)

# print report

predictions = mlp.predict(X_test)
print (classification_report(y_test,predictions))

# save model

filename = 'Poker_NN_model.sav'
pickle.dump(mlp, open(filename,'wb'))



  

        
        