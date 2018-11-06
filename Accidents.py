from ctypes import *
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from scipy import stats, integrate
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import log_loss
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.graph_objs import *
from sklearn import linear_model
from sklearn import metrics
from collections import defaultdict


if __name__ == '__main__':
    df = pd.read_csv('E://Documents//Israel Tech Challenge//fars_train//fars_train.csv')
    df.columns = ['State', "Age", "Gender", "Person Type","Seating Position" ,"RESTRAINT_SYSTEM-USE",
                "AIR_BAG_AVAILABILITY/DEPLOYMENT", "EJECTION", "EJECTION_PATH",  "EXTRICATION", "NON_MOTORIST_LOCATION",
                "POLICE_REPORTED_ALCOHOL_INVOLVEMENT", "METHOD_ALCOHOL_DETERMINATION", "ALCOHOL_TEST_TYPE",
                "ALCOHOL_TEST_RESULT", "REPORTED_DRUG_INVOLVEMENT", "METHOD_OF_DRUG_DETERMINATION",
                "DRUG_TEST_TYPE(1 of 3)","DRUG_TEST_RESULTS (1 of 3)","DRUG_TEST_TYPE (2 of 3)", "DRUG_TEST_RESULTS (2 of 3)",
                "DRUG_TEST_TYPE(3 of 3)","DRUG_TEST_RESULTS (3 of 3)", "HISPANIC_ORIGIN","TAKEN_TO_HOSPITAL","RELATED_FACTOR_(1)-PERSON_LEVEL",
                "RELATED_FACTOR_(2)-PERSON_LEVEL","RELATED_FACTOR_(3)-PERSON_LEVEL",
                "RACE","INJURY_SEVERITY"]

    #first, we shall split the data into training and test, BEFORE having any conclusions
    #(because we want to be as objective as possible):
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]



    # now, let's have a look on how the variables distributed, in order to check wheater or not there are some
    # unnecessary fields:

    # #bar plots and conclusions:
    #
    # f = train['State']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['Age']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['Gender']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['Person Type']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['Seating Position']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['RESTRAINT_SYSTEM-USE']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['AIR_BAG_AVAILABILITY/DEPLOYMENT']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['EJECTION']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['EJECTION_PATH']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['EXTRICATION']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['NON_MOTORIST_LOCATION'] #-----------------CAN BE WITHDRAWED--------------------------------
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=2, align='center')
    #
    # f = train['POLICE_REPORTED_ALCOHOL_INVOLVEMENT']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['METHOD_ALCOHOL_DETERMINATION']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['ALCOHOL_TEST_TYPE']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['ALCOHOL_TEST_RESULT'] #-------------Consider make it as "below allowed" and "over allowed"
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['REPORTED_DRUG_INVOLVEMENT']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['METHOD_OF_DRUG_DETERMINATION']     #-------------Withdraw this one
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')

    # f = train['DRUG_TEST_TYPE(1 of 3)']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['DRUG_TEST_RESULTS (1 of 3)']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['DRUG_TEST_TYPE (2 of 3)']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['DRUG_TEST_RESULTS (2 of 3)']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['DRUG_TEST_TYPE(3 of 3)']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['DRUG_TEST_RESULTS (3 of 3)']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['HISPANIC_ORIGIN']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['TAKEN_TO_HOSPITAL']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['RELATED_FACTOR_(1)-PERSON_LEVEL'] #-----------------------Consider removing this one
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')

    # f = train['RELATED_FACTOR_(2)-PERSON_LEVEL'] #-----------------------Consider removing this one
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['RELATED_FACTOR_(3)-PERSON_LEVEL']  #-----------------------Consider removing this one
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['RACE']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    #
    # f = train['INJURY_SEVERITY']
    # labels, values = zip(*Counter(f).items())
    # values = np.asarray(values)
    # values = values/len(f)
    # plt.bar(labels, values, width=1, align='center')
    # plt.show()

    #We found out that the fields: NON_MOTORIST_LOCATION, METHOD_OF_DRUG_DETERMINATION, RELATED_FACTOR_(1)-PERSON_LEVEL
    #RELATED_FACTOR_(2)-PERSON_LEVEL and RELATED_FACTOR_(3)-PERSON_LEVEL are all have density of one level that got more
    #then 95% percent and the others barely has anything, therefore we shall eliminate these fields

    del train['NON_MOTORIST_LOCATION']
    del train['METHOD_OF_DRUG_DETERMINATION']
    del train['RELATED_FACTOR_(1)-PERSON_LEVEL']
    del train['RELATED_FACTOR_(2)-PERSON_LEVEL']
    del train['RELATED_FACTOR_(3)-PERSON_LEVEL']

    del test['NON_MOTORIST_LOCATION']
    del test['METHOD_OF_DRUG_DETERMINATION']
    del test['RELATED_FACTOR_(1)-PERSON_LEVEL']
    del test['RELATED_FACTOR_(2)-PERSON_LEVEL']
    del test['RELATED_FACTOR_(3)-PERSON_LEVEL']

    del df['NON_MOTORIST_LOCATION']
    del df['METHOD_OF_DRUG_DETERMINATION']
    del df['RELATED_FACTOR_(1)-PERSON_LEVEL']
    del df['RELATED_FACTOR_(2)-PERSON_LEVEL']
    del df['RELATED_FACTOR_(3)-PERSON_LEVEL']

    #Now, let's see if are there any correlations between the numeric features:
    #
    # corr = train.corr()
    # sns.heatmap(corr,
    #             xticklabels=corr.columns.values,
    #             yticklabels=corr.columns.values)
    # plt.show()

    #We can see that 'DRUG_TEST_RESULTS (2 OF 3)' AND 'DRUG_TEST_RESTULTS (3 OF 3)' are highly correlated, therefore we
    #will withdraw the 'DRUG_TEST_RESTULTS (3 OF 3)' feature

    del df['DRUG_TEST_RESULTS (3 of 3)']
    del train['DRUG_TEST_RESULTS (3 of 3)']
    del test['DRUG_TEST_RESULTS (3 of 3)']


    #Let's get the whole sample space in order to encode it equaly between training set and real test results
    sampleSpace = pd.read_csv('E://Documents//Israel Tech Challenge//fars_train//sample_space.csv')
    sampleSpace.columns = ['State', "Age", "Gender", "Person Type", "Seating Position", "RESTRAINT_SYSTEM-USE",
                        "AIR_BAG_AVAILABILITY/DEPLOYMENT", "EJECTION", "EJECTION_PATH", "EXTRICATION",
                        "NON_MOTORIST_LOCATION",
                        "POLICE_REPORTED_ALCOHOL_INVOLVEMENT", "METHOD_ALCOHOL_DETERMINATION", "ALCOHOL_TEST_TYPE",
                        "ALCOHOL_TEST_RESULT", "REPORTED_DRUG_INVOLVEMENT", "METHOD_OF_DRUG_DETERMINATION",
                        "DRUG_TEST_TYPE(1 of 3)", "DRUG_TEST_RESULTS (1 of 3)", "DRUG_TEST_TYPE (2 of 3)",
                        "DRUG_TEST_RESULTS (2 of 3)",
                        "DRUG_TEST_TYPE(3 of 3)", "DRUG_TEST_RESULTS (3 of 3)", "HISPANIC_ORIGIN", "TAKEN_TO_HOSPITAL",
                        "RELATED_FACTOR_(1)-PERSON_LEVEL",
                        "RELATED_FACTOR_(2)-PERSON_LEVEL", "RELATED_FACTOR_(3)-PERSON_LEVEL",
                        "RACE"]

    #add the real test:
    realTest = pd.read_csv('E://Documents//Israel Tech Challenge//fars_train//fars_test.csv')
    realTest.columns = ['State', "Age", "Gender", "Person Type","Seating Position" ,"RESTRAINT_SYSTEM-USE",
                "AIR_BAG_AVAILABILITY/DEPLOYMENT", "EJECTION", "EJECTION_PATH",  "EXTRICATION", "NON_MOTORIST_LOCATION",
                "POLICE_REPORTED_ALCOHOL_INVOLVEMENT", "METHOD_ALCOHOL_DETERMINATION", "ALCOHOL_TEST_TYPE",
                "ALCOHOL_TEST_RESULT", "REPORTED_DRUG_INVOLVEMENT", "METHOD_OF_DRUG_DETERMINATION",
                "DRUG_TEST_TYPE(1 of 3)","DRUG_TEST_RESULTS (1 of 3)","DRUG_TEST_TYPE (2 of 3)", "DRUG_TEST_RESULTS (2 of 3)",
                "DRUG_TEST_TYPE(3 of 3)","DRUG_TEST_RESULTS (3 of 3)", "HISPANIC_ORIGIN","TAKEN_TO_HOSPITAL","RELATED_FACTOR_(1)-PERSON_LEVEL",
                "RELATED_FACTOR_(2)-PERSON_LEVEL","RELATED_FACTOR_(3)-PERSON_LEVEL",
                "RACE"]

    del realTest['NON_MOTORIST_LOCATION']
    del realTest['METHOD_OF_DRUG_DETERMINATION']
    del realTest['RELATED_FACTOR_(1)-PERSON_LEVEL']
    del realTest['RELATED_FACTOR_(2)-PERSON_LEVEL']
    del realTest['RELATED_FACTOR_(3)-PERSON_LEVEL']
    del realTest['DRUG_TEST_RESULTS (3 of 3)']


    del sampleSpace['NON_MOTORIST_LOCATION']
    del sampleSpace['METHOD_OF_DRUG_DETERMINATION']
    del sampleSpace['RELATED_FACTOR_(1)-PERSON_LEVEL']
    del sampleSpace['RELATED_FACTOR_(2)-PERSON_LEVEL']
    del sampleSpace['RELATED_FACTOR_(3)-PERSON_LEVEL']
    del sampleSpace['DRUG_TEST_RESULTS (3 of 3)']


    features = range(0,len(train.columns)-1,1)

    sampleSpace = sampleSpace.iloc[:, features]
    xTrain = train.iloc[:, features]
    yTrain = train.iloc[:, -1]
    xTest = test.iloc[:, features]
    yTest = test.iloc[:, -1]

    #First, we change every categorical feature to an ordinal one:
    X_train_ordinal = xTrain.values
    X_test_ordinal = xTest.values
    Y_train_ordinal1 = yTrain.values
    Y_test_ordinal1 = yTest.values
    realTest_ordinal = realTest.values
    sampleSpace_ordinal = sampleSpace.values
    le = preprocessing.LabelEncoder()
    Y_train_ordinal = le.fit(Y_train_ordinal1)
    Y_train_ordinal = le.transform(Y_train_ordinal1)
    Y_test_ordinal = le.fit(Y_test_ordinal1)
    Y_test_ordinal = le.transform(Y_test_ordinal1)

    les = []
    for i in range(sampleSpace_ordinal.shape[1]):
        if (str(sampleSpace_ordinal[1,i]) != 'nan'):
            cleanedList = [x for x in sampleSpace_ordinal[:,i] if str(x) != 'nan']
            le = preprocessing.LabelEncoder()
            le.fit(cleanedList)
            les.append(le)
            X_train_ordinal[:, i] = le.transform(X_train_ordinal[:, i])
            X_test_ordinal[:, i] = le.transform(X_test_ordinal[:, i])
            realTest_ordinal[:, i] = le.transform(realTest_ordinal[:, i])


    #Now, let's make it dummy variable:
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(X_train_ordinal)
    X_train_one_hot = enc.transform(X_train_ordinal)
    X_test_one_hot = enc.transform(X_test_ordinal)
    realTest_one_hot = enc.transform(realTest_ordinal)

    #Train multinomial logistic regression
    mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(X_train_one_hot, yTrain)


    #Multinomial logistic regression scores:
    print ("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(yTrain, mul_lr.predict(X_train_one_hot)))
    print ("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(yTest, mul_lr.predict(X_test_one_hot)))

    res = mul_lr.predict(realTest_one_hot)
    restoCSV = np.empty(shape=len(res), dtype=int)
    for i in range (0,len(res),1):
        if res[i]=='Possible_Injury':
            res[i]=0
        else:
            if res[i]=='No_Injury':
                res[i]=1
            else:
                if res[i]=='Incapaciting_Injury':
                    res[i]=6
                else:
                    if res[i]=='Fatal_Injury':
                        res[i]=3
                    else:
                        if res[i]=='Unknown':
                            res[i]=4
                        else:
                            if res[i]=='Nonincapaciting_Evident_Injury':
                                res[i]=5
                            else:
                                if res[i]=='Died_Prior_To_Accident':
                                    res[i]=2
                                else:
                                    if res[i]=='Injured_Severity_Unknown':
                                        res[i]=7
        restoCSV[i]=res[i]

    np.savetxt('prediction.csv',restoCSV, delimiter =',')

    # Now we shall activate NN on the data: -------------------------------hasn't been used since he didn't get me better
    # accuracy rates (0.8 on test set, MLR gave me 0.8011 and was much faster)------------------------------------------



    # #create model
    # model = Sequential()
    # #I chose the rectifier function since better performance is achieved using the rectifier activation function
    # model.add(Dense(np.shape(X_train_one_hot)[1], input_dim=np.shape(X_train_one_hot)[1], activation='relu'))
    # model.add(Dense(30, activation='relu'))
    # model.add(Dense(30, activation='relu'))
    # model.add(Dense(30, activation='relu'))
    # model.add(Dense(len(set(Y_train_ordinal)), activation='softmax')) #softmax because we want it to give us an output
    # #of integers
    #
    # # Compile model
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    #
    # # Fit the model
    # #model.fit(X_train_one_hot, Y_train_ordinal, validation_data=(X_test_one_hot,Y_test_ordinal), epochs=15, batch_size=10)
    #
    # # We got the best model when epoching 5 times
    #
    # model.fit(X_train_one_hot, Y_train_ordinal, validation_data=(X_test_one_hot, Y_test_ordinal), epochs=5, batch_size=10)

