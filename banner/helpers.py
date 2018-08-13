import os
from banner._version import __version__

# returns the version number
def getVersion():
	return __version__

# fileCheck makes sure the file exists
def fileCheck(parser, arg):
	if not os.path.exists(arg):
		parser.error("The file {} does not exist!" .format(arg))
	else:
        # return an open file handle
		return open(arg, 'r')
