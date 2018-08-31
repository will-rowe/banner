import numpy as np
import os
import pytest
import banner.rfc as rfc

class TestRCF:
	def test_createRCF(self):
		fh = open("tests/data/AvsB.csv", 'r')
		data = np.loadtxt(fname = fh, delimiter = ',', dtype = int)
		fh.close()
		# split to features and labels
		features = data[:, :-1]
		labels = data[:, -1]
		testRFC = rfc.bannerRFC(features, labels, 1, 100)
		noTrain, noTest = testRFC.prepareData(0.20)
		assert noTrain == 151
		assert noTest == 37
