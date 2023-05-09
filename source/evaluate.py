from multiprocessing.sharedctypes import Value
import warnings
import logging 
import tempfile
from copy import copy
# warnings.filterwarnings('error', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = DeprecationWarning)
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import load_model
from art.utils import to_categorical, load_dataset
import numpy as np
import pandas as pd
from art.config import ART_DATA_PATH
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import ProjectedGradientDescent, AutoAttack, CarliniLInfMethod, CarliniL2Method, FastGradientMethod, PixelAttack, AutoProjectedGradientDescent, DeepFool, HopSkipJump, AdversarialPatch, ThresholdAttack
from art.defences.preprocessor import FeatureSqueezing, GaussianAugmentation, InverseGAN, DefenseGAN, JpegCompression, LabelSmoothing, PixelDefend, Resample, SpatialSmoothing, ThermometerEncoding, TotalVarMin
from art.defences.postprocessor import *
from art.defences.transformer.evasion import *
from scipy.stats import entropy
from art.utils import get_file
from tensorflow.python.framework.ops import disable_eager_execution
from os.path import exists
from os import mkdir, chmod
from sklearn.metrics import precision_score, recall_score

from math import log10, floor
from time import process_time as time
import os

import pickle
import shutil

from hashlib import sha256

from art.utils import get_file, check_and_transform_label_format, compute_accuracy
 #pip install adversarial-robustness-toolbox

disable_eager_execution()

# import numpy as np
# np.seterr(divide='ignore', invalid='ignore')


FOLDER = "./debug/"
VERBOSE = True
OMIT = ['classifier', 'data', 'defense', 'attack', 'adv', 'ben_pred', 'adv_pred']

if not exists(FOLDER):
    mkdir(FOLDER)

def my_hash(x):
    x = str(x).encode('utf-8')
    x = sha256(x).hexdigest()
    x = str(x)
    return x
url = 'https://www.dropbox.com/s/p2nyzne9chcerid/mnist_cnn_original.h5?dl=1'
assert my_hash(url) == '29dfcd90a6957af25455b0db9403d943b14388b2ee29396dfc258d3444c07033'


class Data():
    def __init__(self, X_train, y_train, X_test, y_test, defense = None):
        self.x_train = X_train
        self.y_train = y_train
        self.y_test = y_test
        self.x_test = X_test
        self.defense = defense
        self.id = my_hash(str(vars(self)))
    def __eq__(self, other):
        return my_hash(str(vars(self))) == my_hash(str(vars(other)))

    def __hash__(self, other):
        self.id = my_hash(str(vars(self)))
        return self.id 

#TODO: Port back to load_dataset in art.utils
def load_data(data = "mnist"):
    if data == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        data = Data(X_train, y_train, X_test, y_test)
    elif data == 'cifar10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.reshape(X_train.shape[0], 32, 32, 3).astype('float32') / 255
        X_test = X_test.reshape(X_test.shape[0], 32, 32, 3).astype('float32') / 255
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        data = Data(X_train, y_train, X_test, y_test)
    else:
        logging.info("Dataset not supported")
    return data


class Experiment():
    def __init__(self, data, classifier, classifier_id, defense = None, n = 100, folder = FOLDER):
        assert isinstance(data, Data), "data object must be an instance of Data class"
        self.classifier = classifier
        self.defense = defense
        self.classifier_id = str(my_hash(str([j for i,j in vars(self.classifier).items() if  'at 0x' not in str(j)])+str(type(self.classifier)) +str(self.defense)))
        self.data_id = data.id
        self.attack_id = None
        self.n = n
        if self.defense is not None:
            self.defense_id = str(defense.__dict__) + str(type(defense))
            self.defense_id = str(my_hash(self.defense_id))
        else:
            self.defense_id = str()
        self.id = my_hash(str(self.classifier_id) + str(self.data_id) + str(self.defense_id) + str(self.n))


##TODO fix always verbose train
    def train(self, data, name, verbose = True, is_fitted = False):
        if is_fitted == False: 
            start = time()
            from sklearn.model_selection import train_test_split
            try:
                data.x_train, _, data.y_train, _ = train_test_split(data.x_train, data.y_train, train_size=self.n, stratify = data.y_train)
            except ValueError as e:
                logging.warning(e)
                self.n = data.x_train.shape[0]
            try: # Hopefully the data is in the right format
                self.classifier.fit(data.x_train, data.y_train)
            except TypeError or ValueError as e:
                self.classifier.fit(data.x_train, to_categorical(data.y_train))       
            self.train_time = time()  - start
            self.classifier_name = name
        else:
            self.train_time = np.nan
            self.classifier_name = name
        return self

    def set_attack(self,  name, attack, max_iter = 100, batch_size = 1024, threshold = .3, attack_size = 100, **kwargs):
        logging.info("Attack Type: " + name)
        logging.info("Train Size: " +  str(self.n))
        logging.info("Attack Size: " + str(attack_size))
        adv_attr_list = ["adv", "adv_acc", "adv_rec", "adv_cov", "adv_prec", "adv_pred_time", "attack_time"]
        for attr in adv_attr_list:
            self.__dict__[attr] = np.nan
        self.attack_size = attack_size
        self.attack = attack
        self.attack_name = name
        self.attack_iter = max_iter
        self.attack_batch_size = batch_size
        self.attack_threshold = threshold
        self.attack_id = str(my_hash(str([j for i,j in vars(self.attack).items() if  'at 0x' not in str(j)])+str(type(self.attack))+ str(self.attack_size)))
        self.id = my_hash(str(self.classifier_id) + str(self.data_id) + str(self.defense_id) + str(self.n) + str(self.attack_id) + str(**kwargs))
        logging.info("Attack added Successfully.")
        return self

    def launch_attack(self, data, targeted = False):
        if "Patch" in str(type(self.attack)):
            start = time()
            patches, masks = self.attack.generate(data.x_test[:self.attack_size], data.y_test[:self.attack_size])
            print(self.attack._attack.__dict__.keys())
            self.adv = self.attack.apply_patch(data.x_test[:self.attack_size], scale = self.attack._attack.__dict__['scale_max'])
            self.attack_time = time() - start
        elif targeted == True:
            start = time()
            logging.info("Generating Targeted Attack Samples.")
            self.adv = self.attack.generate(data.x_test[:self.attack_size], data.y_test[:self.attack_size])
            self.attack_time = time() - start
            logging.info("Attack launched Successfully.")
        else: # Catches untargeted attacks
            start = time()
            logging.info("Generating Untargeted Attack Samples.")
            self.adv = self.attack.generate(data.x_test[:self.attack_size])
            self.attack_time = time() - start
            logging.info("Attack launched Successfully.")
        
        return self
        

    def evaluate(self, data, folder, modes = ['accuracy', 'coverage']):
        folder = os.path.join(folder, 'results/')
        logging.info("Transforming data")
        data.y_test   = check_and_transform_label_format(data.y_test)
        other = check_and_transform_label_format(data.y_test, return_one_hot = False)
        if not hasattr(self, 'ben_pred') or not hasattr(self, 'ben_pred_time'):
            sig_figs = int(round(log10(self.n), 0))
            logging.info("Predicting.")
            try:
                start = time()
                self.ben_pred = self.classifier.predict(data.x_test[:10000])
                self.ben_pred_time = time() - start
            except ValueError or TypeError as e:
                logging.info(e)
                start = time()
                self.ben_pred = self.classifier.predict(data.x_test[:10000])
                self.ben_pred_time = self.ben_pred_time = time() - start
            ben_pred = np.argmax(self.ben_pred, axis = 1)
            logging.info("Evaluating.")
            if 'accuracy' in modes:
                self.ben_acc  = compute_accuracy(self.ben_pred, data.y_test[:len(self.ben_pred)])[0]
                logging.info("Benign accuracy: " + str(round(self.ben_acc, sig_figs)))
            if 'coverage' in modes:
                self.ben_cov  = compute_accuracy(self.ben_pred, data.y_test[:len(self.ben_pred)])[1]
                logging.info("Benign coverage: " + str(round(self.ben_cov, sig_figs)))
            if 'precision' in modes:
                self.ben_prec = precision_score(ben_pred, other[:len(ben_pred)], average = None)
                logging.info("Benign precision: " + str(round(self.ben_prec, sig_figs)))
            if 'recall' in modes:
                self.ben_rec  = recall_score(ben_pred, other[:len(ben_pred)], average = None)
                logging.info("Benign recall: " + str(round(self.ben_rec, sig_figs)))
            logging.info("Model evaluated successfully")
            logging.info("Train Time:" + str(round(self.train_time, 2)))
            

        if hasattr(self, 'adv') or (hasattr(self, 'adv_pred') and self.adv_pred == np.nan):
            logging.info("Adversarial Predictions for attack: {}.".format(self.attack_name))
            sig_figs = int(round(log10(self.attack_size), 0))
            start = time()
            self.adv_pred = self.classifier.predict(self.adv)
            self.adv_pred_time = time() - start
            logging.info("Adversarial Evaluations.")
            adv_pred = np.argmax(self.adv_pred, axis = 1)
            if 'accuracy' in modes:
                self.adv_acc  = compute_accuracy(self.adv_pred, data.y_test[:len(self.adv_pred)])[0]
                logging.info("Adversarial accuracy: " + str(round(self.adv_acc, sig_figs)))
            if 'coverage' in modes:
                self.adv_cov  = compute_accuracy(self.adv_pred, data.y_test[:len(self.adv_pred)])[1]
                logging.info("Adversarial coverage: " + str(round(self.adv_cov, sig_figs)))
            if 'precision' in modes:
                self.adv_prec = np.mean(precision_score(adv_pred, other[:len(adv_pred)], average = None))
                logging.info("Adversarial precision: " + str(round(self.adv_prec, sig_figs)))
            if 'recall' in modes:
                self.adv_rec  = np.mean(recall_score(adv_pred, other[:len(adv_pred)], average = None))
                logging.info("Adversarial recall: " + str(round(self.adv_rec, sig_figs)))
            logging.info("Attack Time:" + str(round(self.attack_time, 2)))
            logging.info("Attack evaluated successfully.") 
            sig_figs = int(round(log10(self.attack_size), 0))
            logging.info("Significant Figures: " + str(sig_figs))
            logging.info("Adversarial accuracy: " +  str(round(self.adv_acc, sig_figs)))
            logging.info("Experiment Evaluated Successfully.")
        return self

    def __eq__(self, other):
        truth = self.data_id == other.data_id
        truth = truth and self.attack_id == other.attack_id
        truth = truth and self.defense_id == other.defense_id
        truth = truth and self.data_id == other.data_id
        truth = truth and self.n == other.n
        truth = truth and self.classifier_id == self.classifier_id
        #truth = truth and my_hash(self) == my_hash(other)
        return truth

    def __hash__(self):
        self.id = my_hash(str(self.classifier_id) + str(self.data_id) + str(self.defense_id) + str(self.n) + str(self.attack_size))
        return(self.id)

def generate_model(data, classifier, defense, def_name, train_size, folder = FOLDER, verbose = True):
    experiment_dict = {}
    cl_name = classifier[0]
    classifier = classifier[1]
    data_name = data[0]
    datum = data[1]
    classifier = classifier.__dict__['_model']
    name = str(my_hash(cl_name))
    if  defense is None:
        cl = KerasClassifier(model = classifier)
        is_fitted = True
    elif 'pre' in str(type(defense)):
        cl = KerasClassifier( model=classifier, preprocessing_defences = defense)
        is_fitted = False
    elif 'post' in str(type(defense)):
        cl = KerasClassifier(model = classifier, postprocessing_defences = defense)
        is_fitted = True 
    else:
        logging.info("Defense not supported. Try running the function again, using your defended model as the classifier.")
        raise ValueError
    res = Experiment(datum, cl, name, defense = defense,  n = train_size, folder = folder)
    model_file = folder + res.classifier_id + ".model"
    logging.info("Number of Samples used for Training: "+ str(res.n))
    logging.info("Defense: " +  def_name)
    logging.info("Classifier: " + cl_name)
    logging.info("Data-Name: " +  data_name)
    res.def_name = def_name if def_name is not None else "None"
    res.classifier_name = cl_name
    if exists(model_file):
        try: # Try to load the model
            with open(model_file, 'rb') as file:
                tmp = pickle.load(file)
                logging.info(str(tmp.classifier_id) + " Classifier Exists!")
            res = tmp
        except: # If that doesn't work, train it
            res = res.train(datum, name = cl_name, verbose = verbose, is_fitted = False)
            logging.info("Saving classifier to: "+ model_file)
            with open(model_file, 'wb') as file:
                pickle.dump(res, file)
    else: # Train 
        res = res.train(datum, name = cl_name, verbose = verbose, is_fitted = is_fitted)
        logging.info("Saving classifier to: "+ model_file)
        with open(model_file, 'wb') as file:
            pickle.dump(res, file)
    experiment_dict[res.id] = res
    return experiment_dict


def generate_attacks(experiment, attack, attack_name, data, attack_sizes = [10], folder = FOLDER, omit = OMIT, verbose = True,  result_file = 'results.csv', **kwargs):
    results = {}
    i = 0 
    for attack_size in attack_sizes:
        data_name = data[0]
        datum = data[1]
        if not exists(folder):
            mkdir(folder)
        experiment = experiment.set_attack( attack_name, attack, attack_size = attack_size, **kwargs)
        filename = os.path.join(FOLDER, str(experiment.id) + ".experiment")
        flag = False
        # Check if the experiment has already been run
        if not exists(filename):
            flag = True
        elif exists(filename): #If it has, make sure all the results exist
            logging.info("Experiment ID: {} already exists.".format(experiment.id))
            logging.info("Opening Experiment...")
            with open(filename, 'rb') as file:
                row = pickle.load(file)
            results[experiment.id] = row
            important_results = ['adv_acc','adv_pred_time', 'train_time', 'attack_time', 'attack_size', 'n', 'data_id', 'classifier_id', 'defense_id', 'def_name', 'classifier_name', 'attack_name', 'def_name']
            for key in important_results:
                if key not in row.keys():
                    logging.info("Experiment ID: {} is missing {}. Deteting file and re-running.".format(experiment.id, key))
                    os.remove(filename)
                    del results[experiment.id]
                    flag = True
        else:
            logging.error("Something went wrong during evaluation. You really should not be able to reach this.")
            raise ValueError
        # Run the experiment if it needs to be run
        if flag == True:
            logging.info("Experiment ID: {} started.".format(experiment.id))
            logging.info("Classifier Type: {}".format(experiment.classifier_name))
            logging.info("Defense: "+ experiment.def_name or None)
            attack.__dict__['_estimator'] = experiment.classifier
            experiment = experiment.launch_attack(datum, folder)
            experiment = experiment.evaluate(datum, folder)
            my_keys = experiment.__dict__.keys()
            row = {my_key: experiment.__dict__[my_key] for my_key in my_keys if my_key not in omit}
            results[experiment.id] = row
            with open(filename, 'wb') as file:
                pickle.dump(row, file)
        
    return results

def generate_variable_attacks(attacks,  variables, variable_name, attack_key = None, verbose = True):
    new_attacks = {}
    for attack_name, base_attack in attacks.items():
        if attack_key in attack_name or attack_key == None:
            for variable in variables:
                attack = copy(base_attack)
                attack.__dict__[variable_name] = variable
                new_name = attack_name + "-" + str(variable_name) + ":" + str(variable) 
                new_attacks.update({new_name : attack})
        else:
            attack = copy(base_attack)
            new_attacks.update({attack_name : attack}) # skips attacks that don't have key
    assert len(set(new_attacks.values())) == len(set(new_attacks))
    return new_attacks

def generate_variable_defenses(defenses,  variables, variable_name, defense_key = None):
    new_attacks = generate_variable_attacks(defenses, variables, variable_name, attack_key = defense_key)
    return new_attacks

def append_results(results, folder = FOLDER, filename = 'results.csv'):
    if not exists(folder):
        mkdir(folder)
    df = pd.DataFrame.from_dict(results, orient = 'index')
    df_filename = os.path.join(folder, filename)
    try:
        df_disk = pd.read_csv(df_filename)
    except FileNotFoundError as e:
        logging.warning("No results file found.")
        df_disk = pd.DataFrame()
    try:
        df_disk.drop("Unnamed: 0", axis=1, inplace=True)
        df_disk.drop("Experiment ID", axis=1, inplace=True)
    except:
        pass
    df_all = pd.concat([df_disk, df], sort=True, ignore_index=True)
    df_all.to_csv(df_filename, header = True, mode='w', index = True)
    return df_all

if __name__ == '__main__':
    tmp = tempfile.gettempdir()
    logging.basicConfig(
        level = logging.DEBUG,
        format = '%(asctime)s %(name)s %(levelname)s %(message)s',
        filename=FOLDER +'debug.log',
        filemode = 'w'
    )
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    file_logger = logging.FileHandler(FOLDER+'debug.log')
    file_logger.setLevel(logging.INFO)
    file_logger.setFormatter(formatter)
    stream_logger = logging.StreamHandler()
    stream_logger.setLevel(logging.INFO)
    stream_logger.setFormatter(formatter)
    logger.addHandler(file_logger)
    logger.addHandler(stream_logger)
    

    TRAIN_SIZE = 100
    THRESHOLD = .3
    BATCH_SIZE = 1024
    MAX_ITER = 10

    dx = load_data()
    da = load_data()
    data = {'Default': da}
    dxx = Data(da.x_train, da.y_train, da.x_test[:2], da.y_test[:2])

    # More Robust function testing
    TEST_SIZE = 10
    url = 'https://www.dropbox.com/s/p2nyzne9chcerid/mnist_cnn_original.h5?dl=1'

    path = get_file('mnist_cnn_original.h5', path=ART_DATA_PATH,
                    url=url, extract = True)
    classifier_model = load_model(path)
    keras_classifier = KerasClassifier(model=classifier_model)

    default_classifier_id = my_hash(url)

    attacks = {
            'PGD' : ProjectedGradientDescent(keras_classifier, eps=THRESHOLD, norm = 'inf', eps_step = .1, batch_size = BATCH_SIZE, max_iter = MAX_ITER, targeted = False, num_random_init=False, verbose = VERBOSE),
            'FGM': FastGradientMethod(keras_classifier, eps = THRESHOLD, eps_step = .1, batch_size = BATCH_SIZE), 
            'Carlini': CarliniL2Method(keras_classifier, verbose = VERBOSE, confidence = .99, max_iter = MAX_ITER, batch_size = BATCH_SIZE), 
            'DeepFool': DeepFool(classifier = KerasClassifier(model=classifier_model, clip_values = [0,1]), batch_size = BATCH_SIZE, verbose = VERBOSE), 
            'HopSkipJump': HopSkipJump(keras_classifier, max_iter = MAX_ITER, verbose = VERBOSE),
            'A-PGD' : AutoProjectedGradientDescent(KerasClassifier(model=classifier_model, )),
            'AdversarialPatch': AdversarialPatch(classifier = KerasClassifier(model=classifier_model, clip_values = [0,1]), max_iter = MAX_ITER, verbose = VERBOSE),
            'Threshold Attack': ThresholdAttack(keras_classifier, th = THRESHOLD, verbose = VERBOSE),
            'PixelAttack': PixelAttack(keras_classifier, th = THRESHOLD, verbose = VERBOSE, es = 1), 
            }
    defenses = {"No Defense": None, 
                    "Feature Squeezing": FeatureSqueezing(clip_values = [0,1], bit_depth = 2, apply_predict = True, apply_fit = False),
                    "Gaussian Augmentation": GaussianAugmentation(sigma = .1, apply_predict = True, apply_fit = True, augmentation = False),
                    "Spatial Smoothing": SpatialSmoothing(window_size = 2, apply_predict = True, apply_fit = True),
                    "Label Smoothing": LabelSmoothing(apply_predict = False, apply_fit = True),
                    # "Thermometer": ThermometerEncoding(clip_values = [0,255], num_space = 64, apply_predict = True, apply_fit = True),
                    "Total Variance Minimization": TotalVarMin(apply_predict = True, apply_fit = True),
                    "Class Labels": ClassLabels(apply_predict = False, apply_fit = True),
                    "Gaussian Noise": GaussianNoise(scale = .3, apply_predict = False, apply_fit = True),
                    "Reverse Sigmoid": ReverseSigmoid(),
                    "High Confidence": HighConfidence(apply_predict = True, apply_fit = False),
                    "Rounded": Rounded(apply_predict = True, decimals = 8),
                    }
    classifiers = { default_classifier_id  : keras_classifier,
    #                "Defensive Distillation": DefensiveDistillation(classifier,
                    }
    #TODO Fix broken test here.
    experiments = generate_model(data, classifiers, defenses, train_size = TRAIN_SIZE, folder = FOLDER, verbose = VERBOSE)
    for experiment in experiments.values():
        results = generate_attacks(experiment, attacks, data,  attack_sizes = [1], folder = FOLDER, max_iter = MAX_ITER, batch_size = BATCH_SIZE, threshold = THRESHOLD, verbose = VERBOSE)
        df = append_results(results, folder = FOLDER)
    

    # Basic Object testing
    attack = ProjectedGradientDescent(keras_classifier, eps=.3, eps_step=.1, 
                    max_iter=10, targeted=False, num_random_init=False)


    attack2 = ProjectedGradientDescent(keras_classifier, eps=.3, eps_step=.1, 
                    max_iter=10, targeted=False, num_random_init=False)

    def1 = FeatureSqueezing(clip_values = [0,1], bit_depth = 2, apply_fit = True)
    def2 = FeatureSqueezing(clip_values = [0,1], bit_depth = 2, apply_fit = True)
    def3 = FeatureSqueezing(clip_values = [0,1], bit_depth = 5, apply_fit = True)
    res1 = Experiment(da, keras_classifier, default_classifier_id, def1, n = TEST_SIZE)
    res2 = Experiment(da, keras_classifier, default_classifier_id, def2,  n = TEST_SIZE)
    res3 = Experiment(da, keras_classifier, default_classifier_id, n = TEST_SIZE)
    res4 = Experiment(da, keras_classifier, default_classifier_id,  n = TEST_SIZE)
    res5 = Experiment(da, keras_classifier, default_classifier_id, def3,  n = TEST_SIZE)
    cl1 = KerasClassifier( model=classifier_model)
    cl2 = KerasClassifier( model=classifier_model)
    id1 = str(hash(str(cl1)+str(type(cl2))))
    id2 = str(hash(str(cl2)+str(type(cl1))))
    step_sizes = [.001, .01, .1]
    new_attacks = generate_variable_attacks(attacks, step_sizes, 'step_size', attack_key = 'PGD')
    norms = [1, 2, 'inf']
    new_attacks = generate_variable_attacks(new_attacks, norms, 'norm', attack_key = 'PGD')
    assert new_attacks['PGD-step_size:0.1-norm:1']
    assert da == dx
    assert da.id == dx.id
    assert da is not dxx
    assert res1.attack_id == res2.attack_id
    assert res1.defense_id == res2.defense_id
    assert res1.data_id == res2.data_id
    assert type(res1.defense) != type(res3.defense)
    assert res1 != res5  != res4
    assert res1.classifier_id == res2.classifier_id
    assert res1 != res3
    logging.info(my_hash(res1.id))
    assert res1 == res2
    assert id1 == id2 
    logging.info("ALL TESTS PASSED")
    import gc; gc.collect()
    logging.info("Garbage Collected")
