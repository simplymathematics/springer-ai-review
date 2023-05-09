from numpy import isin
from evaluate import *
from copy import copy
from scipy.stats import zscore
import math 
from scipy.linalg import svdvals
from art.config import set_data_path
from math import ceil
import pandas as pd
import argparse
import collections



# TODO pass verbose to model building
if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Specify Dataset')
	parser.add_argument('--data', metavar ='d', type = str, help = 'Specify either "cifar-10" or "mnist". Other datasets not supported.', default = 'mnist')
	parser.add_argument('--verbose', metavar ='v', type = bool, help = 'Runs the attacks and model building in verbose mode.', default = True)
	parser.add_argument('--batch_size', metavar ='b', type = int, default = 1024)
	parser.add_argument('--max_iter', metavar = 'i', type = int, default = 10)
	parser.add_argument('--threshold', metavar ='t', type = float, default = .03)
	parser.add_argument('--step_size', type = float, default = [.01], nargs = '+')
	parser.add_argument('--train_size', type = int, nargs = "+", metavar = 't', default = [100], help = 'Pass a list of training sizes, separated by a space.')
	parser.add_argument('--attack_size', type = int, nargs = "+", default = [10], help = 'Pass a list of attack sizes, separated by space.')
	args = parser.parse_args()
	in_data = args.data
	train_sizes = args.train_size
	attack_sizes = args.attack_size
	VERBOSE = args.verbose
	BATCH_SIZE = args.batch_size
	MAX_ITER = args.max_iter
	threshold = args.threshold
	step_sizes = args.step_size


	filename = 'experiment.log'
	logging.basicConfig(
		level = logging.DEBUG,
		format = '%(asctime)s %(name)s %(levelname)s %(message)s',
		filename= filename,
		filemode = 'w'
	)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	logger = logging.getLogger('')
	logger.setLevel(logging.INFO)
	file_logger = logging.FileHandler(filename)
	file_logger.setLevel(logging.INFO)
	file_logger.setFormatter(formatter)
	stream_logger = logging.StreamHandler()
	stream_logger.setLevel(logging.DEBUG)
	stream_logger.setFormatter(formatter)
	logger.addHandler(file_logger)
	logger.addHandler(stream_logger)
		
	# TODO: Add support for non-default storage
	
	ART_DATA_PATH= "./models"

	if in_data == 'mnist':
		path = get_file('mnist_cnn_original.h5', extract=False, path=ART_DATA_PATH,
						url="https://www.dropbox.com/s/bv1xwjaf1ov4u7y/mnist_ratio%3D0.h5?dl=1", verbose = True)
		folder = os.path.join(ART_DATA_PATH, 'mnist/')
	elif in_data == 'cifar10':
		path = get_file('cifar-10_cnn_original.h5',extract=False, path=ART_DATA_PATH, url='https://www.dropbox.com/s/hbvua7ynhvara12/cifar-10_ratio%3D0.h5?dl=1', verbose = True)
		folder = os.path.join(ART_DATA_PATH, 'cifar10/')

	da = load_data(in_data)
	ART_DATA_PATH = folder 
	set_data_path(folder)
	classifier_model = load_model(path)
	keras_benign = KerasClassifier( model=classifier_model, use_logits=False)

	data = (in_data, da )
	defenses = {
				"Control": None, 
				"FSQ": FeatureSqueezing(clip_values = [0,255], bit_depth = 4, apply_predict = True, apply_fit = True),
				"Gauss-In": GaussianAugmentation(sigma = .999, augmentation = False),
				"SPS-size:2": SpatialSmoothing(window_size = 2),
				"SPS-size:3": SpatialSmoothing(window_size = 3),
				"SPS-size:4": SpatialSmoothing(window_size = 4),
				"Label": LabelSmoothing(.99),
				"TVM": TotalVarMin(max_iter = MAX_ITER, prob = .01),
				"Gauss-Out": GaussianNoise(scale = 1, apply_fit = True, apply_predict = False),
				"Conf": HighConfidence(.99),
				"Sigmoid": ReverseSigmoid(beta = .01, gamma = 100),
				"Round": Rounded(3),
				}

	classifier = ("Default Classifier", keras_benign)	  

	# TODO
	# Accept arbitrary parameters
	# Type checking a la 3.7?
	bit_depths = [8, 4,]
	defenses = generate_variable_defenses(defenses, bit_depths, 'bit_depth', defense_key = 'FSQ')
	decimals = [1]
	defenses = generate_variable_defenses(defenses, decimals, 'decimals', defense_key = 'Round')
	defenses = generate_variable_defenses(defenses, [.999], 'sigma', defense_key = 'Gauss-In')
	scales = [.1, .5]
	defenses = generate_variable_defenses(defenses, scales, 'scale', defense_key = 'Gauss-Out')
	defenses = generate_variable_defenses(defenses, [.99], 'sigma', defense_key = 'Label')
	gammas = [1]
	betas = [1]
	defenses = generate_variable_defenses(defenses, gammas, 'gamma', defense_key = 'Sigmoid')
	defenses = generate_variable_defenses(defenses, betas, 'beta', defense_key = 'Sigmoid')
	defenses = generate_variable_defenses(defenses, betas, 'beta', defense_key = 'Sigmoid')
	defenses = generate_variable_defenses(defenses, [.001], 'prob', defense_key = 'TVM')
	defenses = generate_variable_defenses(defenses, [.99], 'cutoff', defense_key = 'Conf')
	
	logging.info("Number of Defenses "+ str(len(defenses.values())))

	print("Saving to", folder)
	i = 0
	total = len(train_sizes) * len(defenses)
	for train_size in train_sizes:
		for def_name, defense in defenses.items():
			i += 1
			results = generate_model(data, classifier, defense, def_name, train_size = train_size, folder = os.path.join(folder, "classifiers/"), verbose = VERBOSE)

			logging.info("{} of {}".format(i, total))
	print("===================================================")
	print("=                                                 =")
	print("=                                                 =")
	print("=           Experiment Run Succesfully            =")
	print("=                                                 =")
	print("=                                                 =")
	print("===================================================")
	import gc; gc.collect()
	logging.info("Garbage Collected. Session closed.")
