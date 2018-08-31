#!/usr/bin/env python3
import os
import sys
import time
import argparse
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from banner import helpers
from banner import rfc

class Banner(object):
	"""
	This is the main class to call the program. It parses the user input and launches the request subcommand.
	"""
	def __init__(self):
		parser = argparse.ArgumentParser(
			description='BANNER is a tool that lives inside HULK and aims to make sense of hulk sketches. At the moment, it trains a Random Forest Classifier using a set of labelled hulk sketches. It can then use this model to predict the label of microbiomes as they are sketches by HULK.',
			usage='''banner <subcommand> [<args>]

subcommands:
   train	Train takes a banner-matrix file from hulk and uses it to train a Random Forest Classifier
   predict	Predict collects sketches from STDIN and classifies them using a RFC
   version	Prints the banner version number and exits
''')
		parser.add_argument('subcommand', help='Subcommand to run')
		# grab the subcommand
		args = parser.parse_args(sys.argv[1:2])
		if not hasattr(self, args.subcommand):
			print ("Unrecognized subcommand")
			parser.print_help()
			exit(1)
		# launch the subcommand
		print ("##########\n# BANNER #\n##########\nsubcommand: {}" .format(args.subcommand))
		getattr(self, args.subcommand)()

	"""
	the version method prints the version number and exits
	"""
	def version(self):
		parser = argparse.ArgumentParser(description='Version prints the banner version number and exits')
		print(helpers.getVersion())
		sys.exit(0)

	"""
	the train subcommand takes a matrix of sketches, with the final column containing the labels (ints) and trains a RFC
	"""
	def train(self):
		parser = argparse.ArgumentParser(
			description='Train takes a banner-matrix file from hulk and uses it to train a Random Forest Classifier')
		# this checks the file and returns an open handle
		parser.add_argument('-m', '--matrix', required=True, type=lambda x: helpers.fileCheck(parser, x), help='The matrix from hulk smash')
		parser.add_argument('-o', '--outFile', required=False, default='banner.rfc', help='Where to write the model to')
		parser.add_argument('-p', '--processors', required=False, default=1, help='Number of processors to use for training')
		parser.add_argument('-e', '--estimators', required=False, default=1000, help='Number of estimators to use for training')
		args = parser.parse_args(sys.argv[2:])
		print('loading sketch matrix: {}' .format(args.matrix.name))
		# load the data
		data = np.loadtxt(fname = args.matrix, delimiter = ',', dtype = int)
		args.matrix.close()
		# split to features and labels
		features = data[:, :-1]
		labels = data[:, -1]
		# create the RFC for banner
		print("creating the banner RFC")
		bannerRFC = rfc.bannerRFC(features, labels, int(args.processors), int(args.estimators))
		# split the samples
		print("splitting samples into training and testing sets")
		noTraining, noTesting = bannerRFC.prepareData(0.20)
		print("\tno. training samples: {}\n\tno. testing samples: {}" .format(noTraining, noTesting))
		# run the training
		print("training...")
		print("\ttraining accuracy: {}" .format(bannerRFC.train()))
		# run the testing, print the stats and save the model
		print("testing...")
		bannerRFC.test(args.outFile)
		print("saved model to disk: {}\nfinished.\n##########\n" .format(args.outFile))

	"""
	the predict subcommand collects sketches from STDIN and classifies them using the RFC model
	"""
	def predict(self):
		parser = argparse.ArgumentParser(
			description='Predict collects sketches from STDIN and classifies them using a RFC')
		parser.add_argument('-m', '--model', required=True, help='The model that banner trained')
		parser.add_argument('-p', '--probability', required=False, default=0.90, type=float, help='The probability threshold for reporting classifications')
		parser.add_argument('-v', '--verbose', required=False, default=False, action='store_true', help='Print all predictions and probability, even if threshold not met yet')
		args = parser.parse_args(sys.argv[2:])
		print("loading model: {}" .format(args.model))
		print("waiting for sketches...")
		# load the model
		bannerRFC = pickle.load(open(args.model, 'rb'))
		# wait for input from STDIN
		for line in sys.stdin:
			if args.verbose == True:
				print(" - received a sketch")
			query = np.fromstring(line, sep = ',', dtype = int)
			# get the query into format
			query = np.asarray(query)
			query = np.reshape(query, (1, -1))
			# classify the query
			prediction = bannerRFC.predict(query)
			probability = bannerRFC.predict_proba(query)
			if args.verbose:
				print("\tpredicted label: {}" .format(prediction[0]))
				print("\tlabel probabilities: {} {}" .format(probability[0][0], probability[0][1]))
			# exit if threshold met
			if (probability[0][0] >= args.probability) | (probability[0][1] >= args.probability):
				print("##########\nprobability threshold met!")
				print("predicted label: {}" .format(prediction[0]))
				print("label probabilities: {} {}" .format(probability[0][0], probability[0][1]))
				print("finished.\n##########\n")
				sys.exit(0)
		# if stdin finished but we're still here, no prediction could be made with that probability threshold
		print("could not make prediction within probability threshold ({})!" .format(args.probability))
		print("finished.")
		sys.exit(0)

if __name__ == '__main__':
	try:
		Banner()
	except KeyboardInterrupt:
		print("\ncanceled by user!")
		sys.exit(0)
