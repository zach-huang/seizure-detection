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

    def __init__(self,  vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# ' define shrink function


def f1(dataset):
    dataset = pd.DataFrame(dataset)
    dataset.dropna(axis=0, how='any', inplace=True)
    dataset = abs(dataset - dataset.mean())/(dataset.std())
    dataset_mavg = pd.DataFrame(dataset.rolling(
            window=int(dataset.shape[0]/10),
            min_periods=0,
            center=True).mean())
    return dataset_mavg


# ' clock start
start_time = time.monotonic()


# ' TRAINING DATA
file = 'eegseizurerecord.xlsx'
training_1 = pd.read_excel(file,  sheet_name=0,  header=None,
                           skiprows=None, names=None)

for i in range(0, int(len(training_1)/1012)):
    c = i*1000
    training_1 = training_1.drop(training_1.index[c:c+12])

for i in range(int(int((training_1.shape[1]))/2)):
    training_1.drop(training_1.columns[i], axis=1, inplace=True)

training_1c_names = []

for i in range(int(training_1.shape[1])):
    training_1c_names.append(str(i)+'.')
training_1.columns = training_1c_names
training_2 = pd.DataFrame(index=np.arange(int(training_1.shape[0])),
                          columns=np.arange(int(training_1.shape[1])))

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

Y_0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Y_0 = pd.DataFrame(Y_0)


# ' TESTING DATA
file2 = 'testing.xlsx'
testing_1 = pd.read_excel(file2,
                          sheet_name=0,
                          header=None,
                          skiprows=None,
                          names=None)

for i in range(0,  int(len(testing_1)/1012)):
    c = i*1000
    testing_1 = testing_1.drop(testing_1.index[c:c+12])

for i in range(int(int((testing_1.shape[1]))/2)):
    testing_1.drop(testing_1.columns[i],
                   axis=1,
                   inplace=True)

testing_1c_names = []
for i in range(int(testing_1.shape[1])):
    testing_1c_names.append(str(i)+'.')

testing_1.columns = testing_1c_names

testing_2 = pd.DataFrame(index=np.arange(int(testing_1.shape[0])),
                         columns=np.arange(int(testing_1.shape[1])))

for i in range(int(testing_1.shape[1])):
    df_b2 = f1(testing_1.loc[:, str(i)+'.'])
    testing_2 = pd.concat([testing_2, df_b2],
                          axis=1,
                          ignore_index=True)

testing_2c_names = []
for i in range(0, int(testing_2.shape[1])):
    testing_2c_names.append(str(i)+'.')

testing_2.columns = testing_2c_names
testing_2.dropna(axis=0, how='all', inplace=True)
testing_2.dropna(axis=1, how='all', inplace=True)
testing_2.dropna(axis=0, how='any', inplace=True)

testing_2 = testing_2[::int(testing_2.shape[0]/1000)]
testing_2 = testing_2/float(max(testing_2.max()))

Y_1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Y_1 = pd.DataFrame(Y_1)


# ' TESTING DATA PART II
# ' test data 2
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

# ' cross-validation train-test set

training_2 = training_2.T
training_2 = training_2.reset_index(drop=True)
training = pd.concat([training_2, Y_0], axis=1, ignore_index=True)
print(training.shape)

testing_2 = testing_2.T
testing_2 = testing_2.reset_index(drop=True)
testing = pd.concat([testing_2, Y_1], axis=1, ignore_index=True)
print(testing.shape)

data = pd.concat([training, testing], axis=0, ignore_index=True)
data = data.rename(index=str, columns={1000: 'results'})
print(data.shape)

# ' final test set
testing_4 = testing_4.T
testing_4 = testing_4.reset_index(drop=True)
final_test = testing_4.values.tolist()
final_test = np.asarray(final_test)


for i in range(10):
    data = data.sample(frac=1).reset_index(drop=True)
    train_index = int(0.6*data.shape[0])

    train = data[:train_index]
    Y_train = train.pop('results')
    train = train.values.tolist()
    train = np.asarray(train)

    Y_train = Y_train.T.values.tolist()

    test = data[train_index:data.shape[0]]
    Y_test = test.pop('results')
    test = test.values.tolist()
    test = np.asarray(test)

    Y_test = Y_test.T.values.tolist()

    C_range = np.logspace(-9, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.75, random_state=42)
    grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
    grid.fit(train, Y_train)
    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))
    scores = grid.cv_results_['mean_test_score'].reshape(
            len(C_range), len(gamma_range))

    # ' plotting parameter optimization results
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()

    clf = svm.SVC(C=grid.best_params_['C'],
                  gamma=grid.best_params_['gamma'],
                  kernel='rbf')
    clf.fit(train, Y_train)
    results = abs(Y_test-clf.predict(test))
    print(results)

# ' FINAL TEST
final_results = 100-100*(abs(Y_6-clf.predict(final_test)).sum(axis=None).sum(axis=None))/102
print(clf.predict(final_test))
print(Y_6-clf.predict(final_test))
print(final_results)


# clock end
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
