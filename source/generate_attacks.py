from evaluate import *
import os
from art.config import set_data_path
import argparse
from random import shuffle, seed


#TODO: Run adv patch

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

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Specify Dataset')
	parser.add_argument('--data', metavar ='d', type = str, help = 'Specify either "cifar-10" or "mnist". Other datasets not supported.', default = 'mnist')
	parser.add_argument('--verbose', metavar ='v', type = bool, help = 'Runs the attacks and model building in verbose mode.', default = True)
	parser.add_argument('--batch_size', metavar ='b', type = int, default = 1024)
	parser.add_argument('--max_iter', metavar = 'i', type = int, default = 10)
	parser.add_argument('--threshold', metavar ='t', type = float, default = .03)
	parser.add_argument('--debug', type = bool, default = False)
	parser.add_argument('--step_size', type = float, default = [.01], nargs = '+')
	parser.add_argument('--attack_size', type = int, nargs = "+", default = [10], help = 'Pass a list of attack sizes, separated by space.')
	args = parser.parse_args()
	in_data = args.data
	attack_sizes = args.attack_size
	VERBOSE = args.verbose
	BATCH_SIZE = args.batch_size
	MAX_ITER = args.max_iter
	threshold = args.threshold
	debug = args.debug
	step_sizes = args.step_size
	
	# TODO: Add support for non-default storage
	set_data_path("./models/")
	ART_DATA_PATH = "./models"
	if in_data == 'mnist':
		path = get_file('mnist_cnn_original.h5', extract=False, path=ART_DATA_PATH, url="https://www.dropbox.com/s/bv1xwjaf1ov4u7y/mnist_ratio%3D0.h5?dl=1", verbose = True)
		folder = os.path.join(ART_DATA_PATH, 'mnist/')
	elif in_data == 'cifar10':
		path = get_file('cifar-10_original.h5',extract=False, path=ART_DATA_PATH, url='https://www.dropbox.com/s/hbvua7ynhvara12/cifar-10_ratio%3D0.h5?dl=1', verbose = True)
		folder =  os.path.join(ART_DATA_PATH, 'cifar10/')
	else:
		logging.debug(data + " not supported")
	if debug == True:
		attack_sizes = [1]
	da = load_data(in_data)
	clip_values = (0.0000001,255.0)
	set_data_path(folder)
	ART_DATA_PATH =folder
	# folder = str(in_data) + '/'
	classifier_model = load_model(path)
	keras_benign = KerasClassifier( model=classifier_model, use_logits=False, clip_values = (0.0000001,255.0))
	if debug == True:
		train_sizes = [100]
		folder = 'debug/' + in_data + "/"
		VERBOSE = True
		if not exists('debug'):
			mkdir('debug')
		if not exists(folder):
			mkdir(folder)
	if exists(folder):
		pass
	else:
		mkdir(folder)
	data = (in_data, da )
	classifier = ("Default Classifier", keras_benign)
	# Constructs and Launches all of the attacks

	attacks = {
		'PGD' : ProjectedGradientDescent(keras_benign, eps=threshold, eps_step=.1, batch_size = BATCH_SIZE, max_iter=MAX_ITER, targeted=False, num_random_init=False, verbose = VERBOSE),
		'FGM': FastGradientMethod(keras_benign, eps = threshold, eps_step = .1, batch_size = BATCH_SIZE), 
		'Deep': DeepFool(KerasClassifier( model=classifier_model, use_logits=False, clip_values = clip_values), batch_size = BATCH_SIZE, verbose = VERBOSE, max_iter = MAX_ITER), 
		'Thresh': ThresholdAttack(KerasClassifier( model=classifier_model, use_logits=False, clip_values = clip_values), th = threshold, verbose = VERBOSE, max_iter = MAX_ITER),
		'Pixel': PixelAttack(KerasClassifier( model=classifier_model, use_logits=False, clip_values = clip_values), th = 1, verbose = VERBOSE, max_iter = MAX_ITER), 
		'HSJ': HopSkipJump(keras_benign, max_iter = MAX_ITER, verbose = VERBOSE, init_eval = 10, max_eval = 100),
		'CW': CarliniLInfMethod(keras_benign, verbose = VERBOSE, confidence =  .99, max_iter = MAX_ITER),
		'patch_new': AdversarialPatch(KerasClassifier( model=classifier_model, use_logits=False, clip_values = clip_values), max_iter = MAX_ITER, verbose = VERBOSE, scale_min = .03, scale_max = 1.0),
		}

	# Enable the below for an extended search space for PGD, as an example
	step_sizes = [.001, .01, .1]
	norms = [1, 2, 'inf']
	attacks = generate_variable_attacks(attacks, step_sizes, 'step_size', attack_key = 'PGD')
	attacks = generate_variable_attacks(attacks, norms, 'norm', attack_key = 'PGD')
	attacks = generate_variable_attacks(attacks, step_sizes, 'step_size', attack_key = 'FGM')
	attacks = generate_variable_attacks(attacks, norms, 'norm', attack_key = 'FGM' )
	pixels = [1,2,4,8,16]
	attacks = generate_variable_attacks(attacks, pixels, 'th', attack_key = 'Pixel')
	scales = [.03, .1, .25, .5, 1.0]
	attacks = generate_variable_attacks(attacks, scales, 'scale_max', attack_key = 'Patch')
	logging.info("Number of Attacks: "+ str(len(attacks.values())))
	new_folder = os.path.join(folder, 'classifiers/')
	set_data_path(new_folder)
	logging.info("Reading from " + new_folder)
	if not exists(new_folder):
		mkdir(new_folder)
	files = list(os.listdir(new_folder))
	seed(int(time()))
	shuffle(files)
	i = 0
	print(len(files))
	files = [x for x in files if str(x).endswith(".model")]
	print(len(files))
	input("Press Enter to continue...")
	total = len(files) * len(attacks)
	for filename in files:
		try:
			with open(new_folder + filename, 'rb') as file:
				res = pickle.load(file)
		except EOFError or OSError or IOError:
			i += 1 * len(attacks)
			# old_name = filename
			# new_name = old_name.replace(".model", ".error")
			# os.rename(os.path.join(new_folder, old_name), os.path.join(new_folder, new_name))
			logging.info("Skipping {} due to EOFError".format(filename))
			# continue
		except KeyboardInterrupt:
			old_name = filename
			new_name = old_name.replace(".model", ".interrupt")
			os.rename(os.path.join(new_folder, old_name), os.path.join(new_folder, new_name))
			logging.info("Skipping {} due to KeyboardInterrupt".format(filename))
			break
		else:
			for attack_name, attack in attacks.items():
				logging.info(str(i+1)+" of "+ str(total))
				i += 1
				results = {}
				logging.info("Running " + attack_name)
				try:
					results = generate_attacks(res, attack, attack_name, data, attack_sizes = attack_sizes, folder = os.path.join(folder, "results/"), max_iter = MAX_ITER, batch_size = BATCH_SIZE, threshold = threshold)
				except KeyboardInterrupt as e:
					logging.info("Cancelled by user.")
					break
				try:
					logging.info("Saving " + attack_name)
					results = append_results(results, folder)
					del results
				except KeyboardInterrupt as e:
					logging.info("Cancelled by user.")
					break
			old_name = filename
			new_name = old_name.replace(".model", ".attacked")
			os.rename(os.path.join(new_folder, old_name), os.path.join(new_folder, new_name))
			
					