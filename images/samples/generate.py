import warnings
warnings.filterwarnings('ignore')
from keras.datasets import mnist, cifar10
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import numpy as np

from art.estimators.classification import KerasClassifier
from art.utils import get_file
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
from pathlib import Path
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool, ThresholdAttack, PixelAttack, HopSkipJump, CarliniL2Method, CarliniLInfMethod, AdversarialPatch
from sklearn.model_selection import ParameterGrid
import hashlib
import pickle
from pathlib import Path
from art.utils import compute_success, compute_accuracy
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
mnist_data = (X_train, y_train), (X_test, y_test)
path = "mnist/model.h5"
if not Path(path).exists():
    path = get_file(path,extract=False, path=".",
                    url='https://www.dropbox.com/s/bv1xwjaf1ov4u7y/mnist_ratio%3D0.h5?dl=1')
    model = load_model(path)
    mnist_clf = KerasClassifier(model=model, use_logits=False, clip_values=[0,1])
else:
    model = load_model(path)
    mnist_clf = KerasClassifier(model=model, use_logits=False, clip_values=[0,1])


attacks = {
    "FGM" : { "name" : [FastGradientMethod], "eps": [0.001, .01, .1, .3, .5, 1], "batch_size" :[100]},
    "PGD" : {"name" : [ProjectedGradientDescent], "eps": [0.001, .01, .1, .3, .5, 1], "batch_size" :[100]},
    "CW-10" : {"name" : [CarliniL2Method], "confidence" : [8, 16, 32, 64, 128, 255], "batch_size" :[10], "max_iter" : [100], 'binary_search_steps': [100], "initial_const" : [100]},
    "CW2-10" : {"name" : [CarliniL2Method], "confidence" : [8, 16, 32, 64, 128, 255], "batch_size" :[10], "max_iter" : [100], 'binary_search_steps': [100],"initial_const" : [100]},
    "CWInf-10" : {"name" : [CarliniLInfMethod], "confidence" : [8, 16, 32, 64, 128, 255], "batch_size" :[10], "max_iter" : [100], 'binary_search_steps': [100], "initial_const" : [100]},
    "Thresh" : {"name" : [ThresholdAttack], "th": [8, 16, 32, 64, 128, 255], "max_iter" : [1000]}, 
    "Deep" : {"name" : [DeepFool], "epsilon": [0.001, .01, .1, .3, .5, 1], "batch_size" :[100]},
    "HSJ": {"name" : [HopSkipJump], "max_eval": [10, 30, 50, 80, 100, 1000], "init_eval" : [10], "init_size" : [10]},
    "Patch" : {"name" : [AdversarialPatch], "scale_max" : [.1, .3, .5, .7, .9, 1.0], "scale_min" : [.01]}, #TODO
    "Pixel":  {"name" : [PixelAttack], "th": [1, 2, 4, 8, 16, 32], "max_iter":[10]}, #TODO
    
}



models = {
    "mnist" : mnist_clf,
    # "cifar" : cifar_clf
}

data = {
    "mnist" : mnist_data,
    # "cifar" : cifar_data
}
atk_combinations = {}
if Path("results.pkl").exists():
    with open("results.pkl", "rb") as f:
        results = pickle.load(f)

results = locals().pop("results", {})
for model_name in models:
    results[model_name] = {} if model_name not in results else results[model_name]
    print(f"Running model: {model_name}")
    for key in attacks:
        var_param = max(attacks[key], key=lambda k: len(attacks[key][k]))
        print(f"Running attack: {key} with {len(attacks[key][var_param])} variations for parameter: {var_param}")
        grid = list(ParameterGrid(attacks[key]))
        results[model_name][key] = {} if key not in results[model_name] else results[model_name][key]
        for entry in grid:
            
            entry_without_name = dict(entry)
            entry_without_name.pop("name")
            hash_name = str(hashlib.md5(str(entry_without_name).encode()).hexdigest())
            if hash_name in results[model_name][key]:
                if "params"  in results[model_name][key][hash_name]:
                    if "samples" in results[model_name][key][hash_name]:
                        print(f"Already ran: {entry}")
                        continue
                    else:
                        print(f"Ran but no samples: {entry}")
                        results[model_name][key][hash_name] = {}
                        results[model_name][key][hash_name]['params'] = dict(entry)
                        name = entry.pop("name")
                        model = models[model_name]
                        try:
                            attack = name(model, **entry, verbose=True)
                        except:
                            attack = name(model, **entry)
                        (X_train, y_train), (X_test, y_test) = data[model_name]
                        if "AdversarialPatch" in str(type(attack)):
                            patch, location = attack.generate(X_test[:100], y_test[:100])
                            max_scale = attack._attack.scale_max
                            samples = attack.apply_patch(X_test[:100], scale =max_scale)
                            preds = model.predict(samples)
                            success = compute_success(model, X_test[:100], y_test[:100], samples)
                            results[model_name][key][hash_name]['success'] = success
                            results[model_name][key][hash_name]['samples'] = samples
                        elif "Carlini" in str(type(attack)):
                            samples = attack.generate(X_test[:10], y_test[:10])
                            predictions = model.predict(samples)
                            y_test_cat = to_categorical(y_test[:10], num_classes=10)
                            predictions = predictions - y_test_cat
                            predictions = np.mean(np.amax(predictions, axis=1))
                            results[model_name][key][hash_name]['predictions'] = predictions
                            preds = model.predict(samples)
                            success = compute_success(model, X_test[:10], y_test[:10], samples)
                            results[model_name][key][hash_name]['success'] = success
                            results[model_name][key][hash_name]['samples'] = samples
                        else:
                            print(str(type(attack)))
                            samples = attack.generate(X_test[:100])
                            preds = model.predict(samples)
                            success = compute_success(model, X_test[:100], y_test[:100], samples)
                            results[model_name][key][hash_name]['success'] = success
                            results[model_name][key][hash_name]['samples'] = samples
                else:
                    results[model_name][key][hash_name]['params'] = dict(entry)
            else:
                print(f"Running params: {entry}")
                results[model_name][key][hash_name] = {}
                results[model_name][key][hash_name]['params'] = dict(entry)
                name = entry.pop("name")
                model = models[model_name]
                try:
                    attack = name(model, **entry, verbose=True)
                except:
                    attack = name(model, **entry)
                (X_train, y_train), (X_test, y_test) = data[model_name]
                if "AdversarialPatch" in str(type(attack)):
                    patch, location = attack.generate(X_test[:100], y_test[:100])
                    max_scale = attack._attack.scale_max
                    samples = attack.apply_patch(X_test[:100], scale =max_scale)
                    preds = model.predict(samples)
                    success = compute_success(model, X_test[:100], y_test[:100], samples)
                    results[model_name][key][hash_name]['success'] = success
                    results[model_name][key][hash_name]['samples'] = samples
                elif "Carlini" in str(type(attack)):
                    samples = attack.generate(X_test[:10], y_test[:10])
                    predictions = model.predict(samples)
                    predictions = np.mean(np.amax(predictions, axis=1))
                    results[model_name][key][hash_name]['predictions'] = predictions
                    preds = model.predict(samples)
                    success = compute_success(model, X_test[:10], y_test[:10], samples)
                    results[model_name][key][hash_name]['success'] = success
                    results[model_name][key][hash_name]['samples'] = samples
                else:
                    samples = attack.generate(X_test[:100])
                    preds = model.predict(samples)
                    success = compute_success(model, X_test[:100], y_test[:100], samples)
                    results[model_name][key][hash_name]['success'] = success
                    results[model_name][key][hash_name]['samples'] = samples
                with open("results.pkl", "wb") as f:
                    pickle.dump(results, f)
        with open(f"{model_name}/results_{key}.pkl", "wb") as f:
            pickle.dump(results[model_name][key], f)