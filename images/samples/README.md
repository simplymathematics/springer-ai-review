This folder contains all of the data and code used to generate the images for the `Attacks` section.
- `mnist` contains the collected `.tiff` images for the mnist samples.
- `generate.py` 

To run this:
```
python -m pip install adversarial-robustness-toolbox matplotlib seaborn opencv-python
python generate.py
python plot.py
```
Then execute the respective jupyter notebooks.