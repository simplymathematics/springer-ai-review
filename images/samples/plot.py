
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool, ThresholdAttack, PixelAttack, HopSkipJump, CarliniL2Method, CarliniLInfMethod, AdversarialPatch
attacks = {
    "FGM" : { "name" : [FastGradientMethod], "eps": [0.001, .01, .1, .3, .5, 1], "batch_size" :[100]},
    "PGD" : {"name" : [ProjectedGradientDescent], "eps": [0.001, .01, .1, .3, .5, 1], "batch_size" :[100]},
    "CW" : {"name" : [CarliniL2Method], "confidence" : [8, 16, 32, 64, 128, 255], "batch_size" :[1], "max_iter" : [100], "initial_const" : [1]},
    "CW2" : {"name" : [CarliniL2Method], "confidence" : [8, 16, 32, 64, 128, 255], "batch_size" :[100], "max_iter" : [100], "initial_const" : [1]},
    "CWInf" : {"name" : [CarliniLInfMethod], "confidence" : [8, 16, 32, 64, 128, 255], "batch_size" :[1], "max_iter" : [100], "initial_const" : [1]},
    "Thresh" : {"name" : [ThresholdAttack], "th": [8, 16, 32, 64, 128, 255], "max_iter" : [1000]}, 
    "Deep" : {"name" : [DeepFool], "epsilon": [0.001, .01, .1, .3, .5, 1], "batch_size" :[100]},
    "HSJ": {"name" : [HopSkipJump], "max_eval": [10, 30, 50, 80, 100, 1000], "init_eval" : [10], "init_size" : [10]},
    "Patch" : {"name" : [AdversarialPatch], "scale_max" : [.1, .3, .5, .7, .9, 1.0], "scale_min" : [.01]}, #TODO
    "Pixel":  {"name" : [PixelAttack], "th": [1, 2, 4, 8, 16, 32], "max_iter":[10]}, #TODO
    
}

title_dict = {
    "FGM" : "Fast Gradient Method",
    "PGD" : "Projected Gradient Descent",
    "CW" : "Carlini $\ell_{\infty}$ Method",
    "CW2" : "Carlini $\ell_{2}$ Method",
    "CWInf" : "Carlini $\ell_{\infty}$ Method",
    "Thresh" : "Threshold Attack",
    "Deep" : "DeepFool",
    "HSJ": "Hop Skip Jump",
    "Patch" : "Adversarial Patch",
    "Pixel":  "Pixel Attack",
}


if Path("results.pkl").exists():
    with open("results.pkl", "rb") as f:
        results = pickle.load(f)
results = locals().pop("results", {})

plt.rcParams["font.family"] = "Times New Roman"
data = "mnist"
mnist_results = results[data]
res = mnist_results
res = results[data]
for key in res:
    if key not in title_dict:
        continue
    if key not in attacks:
        continue
    title = title_dict[key]
    length = len(res[key])
    if length == 0:
        continue
    fig, axs = plt.subplots(1, length)
    i = 0
    
    var_param = max(attacks[key], key=lambda k: len(attacks[key][k]))
    res[key] = {k: v for k, v in sorted(res[key].items(), key=lambda item: item[1]['params'][var_param])}
    print(f"Plotting attack, {key}, with {len(res[key])} variations of parameter: {var_param}")
    # print("!"*80)
    # print(f"Attack: {key} with {len(res[key][var_param])} variations for parameter: {var_param}")
    # print("!"*80)
    for sub_key in res[key]:
        
        sample = res[key][sub_key]['samples'][0]
        if data == "mnist":
            cmap = "gray"
        else:
            cmap = "hot"
        if sample.shape[-1] != 1:
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        if length == 1:
            ax = axs.imshow(sample, cmap=cmap)
        else:
            ax = axs[i].imshow(sample, cmap=cmap)
        ax.set_cmap('gray')
        # ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        print(res[key][sub_key]['params'])
        param_name = var_param
        print(f"Key: {param_name} for model: {key}")
        
        if param_name == "epsilon" or "eps":
            if key == "FGM" or key == "PGD" or key == "Deep":
                new_param_name = "$\| \epsilon \|_2$"
            else:
                new_param_name = "$\epsilon$"
        if param_name == "th":
            if key == "Thresh":
                
                new_param_name = "$\| \epsilon \|_{\infty}$"
                num = "{0:.2f}".format(res[key][sub_key]['params'][param_name]/255 *100)
            else:
                new_param_name = "$\| \epsilon \|_0$"
                total_resolution = sample.size
                num = "{0:.2f}".format(res[key][sub_key]['params'][param_name]/total_resolution * 100)
            num = round(float(num))
            res[key][sub_key]['params'][param_name] = f"{num}%"
        if param_name == "confidence":
            new_param_name = "$C$"
            # num = "{0:.2f}".format(res[key][sub_key]['params'][param_name]/255 * 100)
            num = res[key][sub_key]['params'][param_name]
            num = round(num)
            res[key][sub_key]['params'][param_name] = f"{num}"
        if param_name == "max_eval":
            new_param_name = "$Q$"
        if param_name == "scale_max":
            new_param_name = "$\| \epsilon \|_{0}$"
            num = "{0:2.2f}".format(res[key][sub_key]['params'][param_name] * 100)
            num = round(float(num))
            res[key][sub_key]['params'][param_name] = f"{num}%"
        if "new_param_name" not in locals():
            new_param_name = param_name
        if "scorer" not in locals():
            scorer = "$\eta$"
        if "score" not in locals():
            score = res[key][sub_key]['success']
        if i == 0:
            ax.axes.set_title(fr"{new_param_name} = {res[key][sub_key]['params'][param_name]}")
            ax.axes.set_xlabel(f"{scorer} = {score}")
            ax.axes.set_xticks([])
        else:
            ax.axes.set_title(fr"{res[key][sub_key]['params'][param_name]}")
            ax.axes.set_xlabel(f"{score}")
            ax.axes.set_xticks([])
        del new_param_name
        del scorer
        del score
        # print(f" New param name: {new_param_name} for {param_name} = {res[key][sub_key]['params'][param_name]}")
        # input("Press enter to continue")
        i +=1
    fig.set_size_inches(7.5, 2)
    fig.set_dpi(1000)
    fig.suptitle(title, y =.90)
    fig.tight_layout()
    fig.savefig(f"{data}/{key}.pdf", format="pdf", dpi=1000)