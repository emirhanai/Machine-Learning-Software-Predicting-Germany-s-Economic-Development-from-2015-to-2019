#Library is importing :)
import pandas as pd
from sklearn.ensemble import *
from sklearn.model_selection import *
from sklearn.metrics import *

#Datasets Loadinggg :)
      #df = pd.read_csv('germany_data.csv')
df = pd.read_csv('germany_data_change.csv')

class0 = "2015 values > 2019 values. Maybe, Economy parameter is very little good.."
class1 = "2019 values > 2015 values! Very Good.. Economy parameter is very good."



#features samples
X = df.iloc[:,1:6].values
      #features samples
      #X = df.iloc[:,1:22].values
      #labels,target,values samples
      #According of the Class; Industrial Production (annual variation in %).
      #0 mean [Industrial Production (annual variation in %) < 1]
      #1 mean [Industrial Production (annual variation in %) > 1]
      #y = df.loc[:,['Class']].values
# labels,target,values samples
# According of the Class; It represents the rate of change in 2015 compared to 2019.
# 0 mean [2015 values > 2019 values]
# 1 mean [2015 values < 2019 values]
y = df.loc[:,['Class']].values

#print(X.shape)
#print(y.shape)

#For Loop is start...
for i in range(1,2):

    # X_train,X_test,y_train,y_test creating :)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.24, shuffle=True, stratify=None,
                                                        random_state=55)

    # model creation
    from sklearn.tree import *
    model_germany = ExtraTreesClassifier(n_estimators=1,criterion="gini", max_features="auto", random_state=12)

    # model fitting :)
    model_germany.fit(X_train, y_train)

    # create of prediction with X_test data and model :)
    prediction = model_germany.predict(X_test)

    # Accuracy
    #print("X",i)
    print(f"Accuracy: {accuracy_score(y_test, prediction)*100} ")

    # Confusion Matrix
    print(f"Confusion Matrix: {confusion_matrix(y_test, prediction)*100} ")

    # Precision
    print(f"Precision: {precision_score(y_test,prediction)*100}")

    # Recall
    print(f"Recall: {recall_score(y_test,prediction)*100}")

    # F1
    print(f"F1: {f1_score(y_test,prediction)*100}")

    #Random prediction
    predict = model_germany.predict([[1.8,3.8,2.5,3.5,0.6]])
    #(predict)

    #prediction of model creating
    model_regres = ExtraTreeRegressor(criterion="mse",splitter="random")

    #Regress is prediction model, the model fitting in Below
    model_regres.fit(X_train,y_train)

    #Prediction Model is the Accuracy Score :)<<>>
    print("Prediction is Accuracy: ",model_regres.score(X_test,y_test)*100)



    def prediction_function():

        # Prediction Values
        predics = model_regres.predict([[1.8, 3.8, 2.5, 3.5, 0.6]])

        if predics == [0]:
            return f"Prediction economy class: ['{class0}']"
        else:
            return f"Prediction economy class: ['{class1}']"

    print(prediction_function())


    #print of predictin class


    #import of plt method in pyplot, matplotlib library
    import matplotlib.pyplot as plt


    #graph :)
    plt.scatter(X[12:13],X[6:7])
    plt.legend(["Economic Growth\nIndustrial Production"])
    plt.xlabel('Industrial Production (annual variation in %)')
    plt.ylabel('Economic Growth (GDP - annual variation in %)')
    plt.title("The table that rates the share of \n industrial production in economic growth.")
    #plt.show()

    #importing of the ExportGraphiz function at tree method in Sklearn library :)
    from sklearn.tree import export_graphviz

    #feature names creating
    feature_names = df.iloc[:,1:6].columns

    #target names creating
    target_names = df.loc[:,['Class']].columns

    #save of classification to .dot :)

    def save_decision_trees_as_dot(model_germany, iteration, feature_name):
        file_name = open("emirhan_germany_machine-learning" + str(iteration) + ".dot", 'w')
        dot_data = export_graphviz(
            model_germany,
            out_file=file_name,
            feature_names=feature_name,
            class_names=['2015 values > 2019 values','2019 values > 2015 values'],
            rounded=True,
            proportion=False,
            precision=2,
            filled=True, )
        file_name.close()
        print("Extra Tree in forest :) {} saved as dot file".format(iteration + 1))


    #Save of the .dot loop..
    for i in range(len(model_germany.estimators_)):
        save_decision_trees_as_dot(model_germany.estimators_[i], i, feature_names)
        print(i)