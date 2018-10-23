# Module file for implementation of ID3 algorithm.

# You can add optional keyword parameters to anything, but the original
# interface must work with the original test file.
# You will of course remove the "pass".

import os, sys, dill
import numpy as np
import pandas as pd
import sklearn.metrics as skm
from nltk import FreqDist
# You can add any other imports you need.

class DecisionTree:
    def __init__(self, load_from=None):
        self.tree = {}
        # Fill in any initialization information you might need.
        #
        # If load_from isn't None, then it should be a file *object*,
        # not necessarily a name. (For example, it has been created with
        # open().)
        print("Initializing classifier.")
        if load_from is not None:
            print("Loading from file object.")
            model_instance = dill.load(load_from)
            self.tree = model_instance.tree

    
    def find_information_gain_max(self, X, y, attrs):
        vcount = y.value_counts()/len(y)
        EntropyS = - np.sum(vcount * np.log2(vcount))
        GainDict = {}
        for column in attrs:
            Gain = EntropyS
            for vlu in set(X[column]):              
                idx_of_ftr_vlu = X.index[X[column] == vlu].tolist()
                class_of_vlu = y.loc[idx_of_ftr_vlu]
                class_vcount = class_of_vlu.value_counts()/len(class_of_vlu)
                EntTimesProb = len(class_of_vlu)/len(y) * (- np.sum(class_vcount * np.log2(class_vcount)))
                Gain = Gain - EntTimesProb
            GainDict[column] = Gain

        v=list(GainDict.values())
        k=list(GainDict.keys())
        IG_max_column = k[v.index(max(v))]
        return (IG_max_column)

    def split_parent_to_children(self, X, y, column_name):
        frames = []
        correct_class_for_frame = []
        feature_values = []
        for vlu in set(X[column_name]):
            idx_of_ftr_vlu = X.index[X[column_name] == vlu].tolist()
            frames.append(X.loc[idx_of_ftr_vlu])
            correct_class_for_frame.append((y.loc[idx_of_ftr_vlu]))
            feature_values.append(vlu)

        return frames, correct_class_for_frame, feature_values

    def build_tree(self, X, y, attrs, prune=False):
        if not X.empty:
            IG_MAX = DecisionTree.find_information_gain_max(self, X, y, attrs)

            frames, classes, feature_values = self.split_parent_to_children(X, y, IG_MAX)
        if len(set(y)) == 1 or len(attrs) == 0:

            return y.mode()[0]
        else:
            tree = {IG_MAX:{}}
            for i in range (len(frames)):
                frame = frames[i].drop(IG_MAX, axis = 1)
                subtree = self.build_tree(frame, classes[i], frame.columns)
                tree[IG_MAX][feature_values[i]] = subtree

            return tree
    def train(self, X, y, attrs, prune=False):
        self.tree = DecisionTree.build_tree(self, X, y, attrs)
        # Doesn't return anything but rather trains a model via ID3
        # and stores the model result in the instance.
        # X is the training data, y are the corresponding classes the
        # same way "fit" worked on SVC classifier in scikit-learn.
        # attrs represents the attribute names in columns order in X.
        #
        # Implementing pruning is a bonus question, to be tested by
        # setting prune=True.
        #
        # Another bonus question is continuously-valued data. If you try this
        # you will need to modify predict and test.
    
    def predict(self, instance, tree):
        if tree == {}:
            raise ValueError('ID3 model is not trained') 
        column_to_split_on = next(iter(tree))

        frame_split_on_column = tree[column_to_split_on]

        value_for_col_in_instance = instance[column_to_split_on].iloc[0]

        if value_for_col_in_instance in frame_split_on_column:
            new_dataframe = frame_split_on_column[value_for_col_in_instance]
        else:
            new_dataframe = FreqDist(frame_split_on_column).most_common(1)[0][1]

        if type(new_dataframe) is not dict:
            return new_dataframe
        else:
            return (self.predict(instance, new_dataframe))
        # Returns the class of a given instance.
        # Raise a ValueError if the class is not trained.

    def test(self, X, y, display=False):
        rows = [X[i:i+1] for i in range(len(X))]

        predictions = [self.predict(row, self.tree) for row in rows]
        
        accuracy = skm.accuracy_score(y, predictions)
        
        recall = skm.recall_score(y, predictions, average = 'macro')

        precision = skm.precision_score(y, predictions, average = 'macro')

        f1 = skm.f1_score(y, predictions, average = 'macro')

        cf = skm.confusion_matrix(y, predictions, labels = list(set(y)))

        # Returns a dictionary containing test statistics:
        # accuracy, recall, precision, F1-measure, and a confusion matrix.
        # If display=True, print the information to the console.
        # Raise a ValueError if the class is not trained.
        result = {'precision':precision,
                  'recall':recall,
                  'accuracy':accuracy,
                  'F1':f1,
                  'confusion-matrix':cf}
        if display:

            print(result)
        return (predictions)

    def __str__(self):
        # Returns a readable string representation of the trained
        # decision tree or "ID3 untrained" if the model is not trained.
        if self.tree == {}:
            return "ID3 untrained"
        else:
            string = "DECISION TREE AS A NESTED DICTIONARY:" + "\n" *2 + str(self.tree)
            return string

    def save(self, output):
        dill.dump(self, output)

        # 'output' is a file *object* (NOT necessarily a filename)
        # to which you will save the model in a manner that it can be
        # loaded into a new DecisionTree instance.
