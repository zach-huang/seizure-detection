import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import timedelta
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

#graph
def f2(dataset):
    dataset = pd.DataFrame(dataset)
    dataset.plot()
    plt.show()
"""
---------------------CODE BEGINS BELOW-------------------------------------------------------------------------------
"""



#clock start
start_time = time.monotonic()




"""
DATA COLLECTION BEGINS -----------------------------------------------------------------------------
"""



#TRAINING DATA
file = 'eegseizurerecord.xlsx'
training_1 = pd.read_excel(file,sheet_name=0,header=None,skiprows=None,names=None)
for i in range(0,int(len(training_1)/1012)):
    c = i*1000
    training_1 = training_1.drop(training_1.index[c:c+12])

for i in range(int(int((training_1.shape[1]))/2)):
    training_1.drop(training_1.columns[i],axis=1, inplace=True)

training_1c_names = []
for i in range(int(training_1.shape[1])):
    training_1c_names.append(str(i)+'.')
training_1.columns = training_1c_names

training_1.dropna(axis=0,how='any',inplace=True)
training_1.dropna(axis=1,how='all',inplace=True)

training_1 = abs(training_1 - training_1.mean())/(training_1.std())
training_1 = pd.DataFrame(training_1.rolling(window=int(training_1.shape[0]/10), min_periods=0, center=True).mean())
training_1 = training_1/float(max(training_1.max()))

training_2 = pd.DataFrame(index=np.arange(int(training_1.shape[0])),columns=np.arange(int(training_1.shape[1])))
for i in range(int(training_1.shape[1])):
    df_b1 = f2(training_1.loc[:,str(i)+'.'])
    training_2 = pd.concat([training_2,df_b1],axis=1,ignore_index=True)
training_1 = training_1[::int(training_1.shape[0]/1000)]
training_2 = training_2.T
training = training_2.values.tolist()
training = np.asarray(training)
print(training.shape)
Y = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

#TESTING DATA
file2 = 'testing.xlsx'
testing_1 = pd.read_excel(file2,sheet_name=0,header=None,skiprows=None,names=None)
for i in range(0,int(len(testing_1)/1012)):
    c = i*1000
    testing_1 = testing_1.drop(testing_1.index[c:c+12])

for i in range(int(int((testing_1.shape[1]))/2)):
    testing_1.drop(testing_1.columns[i],axis=1, inplace=True)

testing_1c_names = []
for i in range(int(testing_1.shape[1])):
    testing_1c_names.append(str(i)+'.')
testing_1.columns = testing_1c_names

testing_1.dropna(axis=0,how='any',inplace=True)
testing_1.dropna(axis=1,how='all',inplace=True)
testing_1 = testing_1[::int(testing_1.shape[0]/1000)]

testing_1 = abs(testing_1 - testing_1.mean())/(testing_1.std())
testing_1 = pd.DataFrame(testing_1.rolling(window=int(testing_1.shape[0]/10), min_periods=0, center=True).mean())
testing_1 = testing_1/float(max(testing_1.max()))

testing_2 = pd.DataFrame(index=np.arange(int(testing_1.shape[0])),columns=np.arange(int(testing_1.shape[1])))
for i in range(int(testing_1.shape[1])):
    df_b1 = f2(testing_1.loc[:,str(i)+'.'])
    testing_2 = pd.concat([testing_2,df_b1],axis=1,ignore_index=True)
testing_2 = testing_2.T
testing = testing_2.values.tolist()
testing = np.asarray(testing)
print(testing.shape)






"""
DATA COLLECTION COMPLETE-----------------------------------------------------------------------------
"""







#svm gamma & c parameter optimization
C_range = np.logspace(-9, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.75, random_state=42)
grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
grid.fit(training, Y)
print("The best parameters are %s with a score of %0.2f"% (grid.best_params_, grid.best_score_))
scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),len(gamma_range))

#plotting parameter optimization results
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()

#svm fitting
clf = svm.SVC(C=grid.best_params_['C'],gamma=grid.best_params_['gamma'], kernel='rbf')
clf.fit(training,Y)
fail_results = clf.predict(training)
print(fail_results)
results = clf.predict(testing)
print(results)

#clock end
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))