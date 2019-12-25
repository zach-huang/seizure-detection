import pandas as pd
import numpy as np
import time
from datetime import timedelta
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from multiprocessing import Pool, cpu_count







def f1(dataset):
    dataset = pd.DataFrame(dataset)
    dataset.dropna(axis=0, how='any', inplace=True)
    dataset = abs(dataset - dataset.mean())/(dataset.std())
    dataset_mavg = pd.DataFrame(dataset.rolling(
            window=int(dataset.shape[0]/10),
            min_periods=0,
            center=True).mean())
    dataset_final = dataset_mavg[::int((dataset_mavg.shape(0))/1000)]

    return list(dataset_final)


def process_Pandas_data(func, df, num_processes=None):
    ''' Apply a function separately to each column in a dataframe, in parallel.'''

    # If num_processes is not specified, default to minimum(#columns, #machine-cores)
    if num_processes==None:
        num_processes = min(df.shape[1], cpu_count())

    # 'with' context manager takes care of pool.close() and pool.join() for us
    with Pool(num_processes) as pool:

        # we need a sequence of columns to pass pool.map
        seq = [df[col_name] for col_name in df.columns]

        # pool.map returns results as a list
        results_list = pool.map(func, seq)

        # return list of processed columns, concatenated together as a new dataframe
        return pd.concat(results_list, axis=1)








# ' clock start
start_time = time.monotonic()

# ' LOADING TRAINING DATA
file = 'trainingdataset.xlsx'


training_1 = pd.read_excel(file,
                           sheet_name=0,  header=None,
                           skiprows=None, names=None)


# ' FORMATTING/DATA CLEANING
for i in range(0, int(len(training_1)/1012)):
    c = i*1000
    training_1 = training_1.drop(training_1.index[c:c+12])

for i in range(int(int((training_1.shape[1]))/2)):
    training_1.drop(training_1.columns[i], axis=1, inplace=True)


training_2 = process_Pandas_data(f1, training_1)



"""
for i in range(int(training_1.shape[1])):
    df_b1 = f1(training_1.loc[:, str(i)+'.'])
    training_2 = pd.concat([training_2, df_b1],
                           axis=1,
                           ignore_index=True)

training_2c_names = []

for i in range(0, int(training_2.shape[1])):
    training_2c_names.append(str(i)+'.')
training_2.columns = training_2c_names
training_2.dropna(axis=0, how='all', inplace=True)
training_2.dropna(axis=1, how='all', inplace=True)
training_2.dropna(axis=0, how='any', inplace=True)

training_2 = training_2[::int(training_2.shape[0]/1000)]
training_2 = training_2/float(max(training_2.max()))
"""


Y_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

data = training_2.T



for i in range(5):
    bestC = 0
    bestgamma = 0
    bestscore = 0
    C_range = np.logspace(-9, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.75, random_state=42)
    grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
    grid.fit(data, Y_train)

    if grid.best_score_> bestscore:
        bestscore = grid.best_score_
        bestC = grid.best_params_['C']
        bestgamma = grid.best_params_['gamma']

clf = svm.SVC(C=bestC, gamma=bestgamma, kernel='rbf')









"""""
# ' TESTING DATA
# ' test data
file3 = 'testingcute.xlsx'
testing_3 = pd.read_excel(file3, header=None, skiprows=None, names=None)
testing_3c_names = []
for i in range(int(testing_3.shape[1])):
    testing_3c_names.append(str(i)+'.')
testing_3.columns = testing_3c_names
testing_4 = pd.DataFrame(index=np.arange(int(testing_3.shape[0])),
                         columns=np.arange(int(testing_3.shape[1])))
for i in range(int(testing_3.shape[1])):
    df_b3 = f1(testing_3.loc[:, str(i)+'.'])
    testing_4 = pd.concat([testing_4, df_b3], axis=1, ignore_index=True)
testing_4c_names = []
for i in range(0, int(testing_4.shape[1])):
    testing_4c_names.append(str(i)+'.')
testing_4.columns = testing_4c_names
testing_4.dropna(axis=0, how='all', inplace=True)
testing_4.dropna(axis=1, how='all', inplace=True)
testing_4.dropna(axis=0, how='any', inplace=True)

testing_4 = testing_4[::int(testing_4.shape[0]/1000)]
testing_4 = testing_4/float(max(testing_4.max()))

Y_2 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
Y_3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
Y_4 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
       0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
Y_5 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
Y_6 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


# ' final test set
testing_4 = testing_4.T
testing_4 = testing_4.reset_index(drop=True)
final_test = testing_4.values.tolist()
final_test = np.asarray(final_test)

"""




"""
# ' FINAL TEST
final_results = 100-100*(abs(Y_6-clf.predict(final_test)).sum(axis=None).sum(axis=None))/len(Y_6)
print(clf.predict(final_test))
print(Y_6-clf.predict(final_test))
print(final_results)


# clock end
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
"""