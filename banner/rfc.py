import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class bannerRFC(object):
    """
    This is the Random Forest Class that will train, test and save the model for banner
    """
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    # splits the data into train and test sets, returns the number of samples in each
    def prepareData(self, testProportion):
        self.testProportion = testProportion
        self.trainFeatures, self.trainLabels, self.testFeatures, self.testLabels = splitData(self.features, self.labels, self.testProportion)
        return (len(self.trainLabels), len(self.testLabels))

    # trains the RFC and returns the accuracy for the training
    def train(self):
    	self.RFC=RandomForestClassifier(bootstrap=True, class_weight=None, n_estimators=1000, n_jobs=1)
    	self.RFC.fit(self.trainFeatures, self.trainLabels)
    	trainPredictions = self.RFC.predict(self.trainFeatures)
    	return(accuracy_score(self.trainLabels, trainPredictions))

    # tests the RFC, prints the stats and saves the model
    def test(self, outfile):
    	testPredictions = self.RFC.predict(self.testFeatures)
    	print("f1 score: {}" .format(f1_score(self.testLabels, testPredictions, average="macro")))
    	print("precision: {}" .format(precision_score(self.testLabels, testPredictions, average="macro")))
    	print("recall score: {}" .format(recall_score(self.testLabels, testPredictions, average="macro")))
    	print("test accuracy: {}" .format(accuracy_score(self.testLabels, testPredictions)))
    	# save the model to disk
    	pickle.dump(self.RFC, open(outfile, 'wb'))

# function splitData splits input data into training and testing sets
def splitData(features, labels, test_size):
	total_test_size = int(len(features) * test_size)
	np.random.seed(2)
	indices = np.random.permutation(len(features))
	train_features = features[indices[:-total_test_size]]
	train_labels = labels[indices[:-total_test_size]]
	test_features  = features[indices[-total_test_size:]]
	test_labels  = labels[indices[-total_test_size:]]
	return train_features, train_labels, test_features, test_labels
