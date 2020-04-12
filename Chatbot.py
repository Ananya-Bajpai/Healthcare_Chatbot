import numpy as np
import pandas as pd
from sklearn import *
from sklearn.tree import *
from sklearn.tree import _tree
from sklearn.model_selection import *

# Training and Testing files contain predefined diseases along with symptomps
train = pd.read_csv('E:/Python/Training.csv')
test  = pd.read_csv('E:/Python/Testing.csv')

column = train.columns
column = column[:-1]
a = train[column]
b = train['prognosis']
data = train.groupby(train['prognosis']).max()

#mapping strings to numbers present in the csv files
encoder = preprocessing.LabelEncoder()
encoder.fit(b)
b = encoder.transform(b)

a_train, a_test, b_train, b_test = train_test_split(a, b)
test_a    = test[column]
test_b    = test['prognosis']  
test_b    = encoder.transform(test_b)

classifier1  = DecisionTreeClassifier()
classifier = classifier1.fit(a_train,b_train)
importance = classifier.feature_importances_
index = np.argsort(importance)[::-1]
features = column

print("\nEnter Yes if you have the symptoms and No if you do not have the symptoms") 
def probable_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = encoder.inverse_transform(val[0])
    return disease
def tree(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
        ]
    symptoms_present = []
    def diseases(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            disease_name = feature_name[node]
            threshold = tree_.threshold[node]
            print("\n"+ disease_name + "?\n")
            answer = input()
            # Asking the user if they have the symptoms displayed
            if answer == 'Yes':
                val = 1
            else:
                val = 0
            if  val <= threshold:
                diseases(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(disease_name)
                diseases(tree_.children_right[node], depth + 1)
        else:
            present_disease = probable_disease(tree_.value[node])
            print( "Possible disease: " +  present_disease )
            red_column = data.columns 
            symptoms_given = red_column[data.loc[present_disease].values[0].nonzero()]
            print("\nPresent symtomp:  " + str(list(symptoms_present)))
            print("\nKnown symptomps of the disease: "  +  str(list(symptoms_given)) )  
    diseases(0, 1)

tree(classifier,column)

