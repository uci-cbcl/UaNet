[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

# Clinically Applicable Deep Learning Framework for Organs at Risk Delineation in CT images

## License

Copyright (C) 2019 University of California Irvine and DEEPVOXEL Inc.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

**Note: The code/software is licensed for non-commerical academic research purpose only.**

## Reference
If you use the code or data in your research, we will appreciate it if you could cite the following paper:
```
Tang et al, Clinically applicable deep learning framework for organs at risk delineation in CT images

Nature Machine Intelligence, 1, pages 480–491 (2019)
```

## Data

* Images, annotations and preprocessed files for dataset2 and dataset3 are freely available for non-commercial research pursposes at [here](<https://drive.google.com/drive/folders/15qQFagHnU-1ILNtN3pWeAhCoimfTN9tL?usp=sharing>).
* The original dicom images for dataset2 are freely available at [Head-Neck Cetuximab](<http://doi.org/10.7937/K9/TCIA.2015.7AKGJUPZ>) and [Head-Neck-PET-CT](<https://doi.org/10.7937/K9/TCIA.2017.8oje5q00>). 
* The original images and annotations for dataset3 are freely available at [PDDCA](<http://www.imagenglab.com/newsite/pddca/>).

* Use [this link](<http://irvine.deep-voxel.com:9995/>) to request a copy of the test data of dataset1.

Once you download the data, unzip them and put them under data/raw and data/preprocessed.

## Trained models

* Use [this link](<http://irvine.deep-voxel.com:9995/request>) to request pre-trained model checkpoints for non-commercial academic research purposes. 

Once you download the model checkpoints, change the config['initial_checkpoint'] to the path of the file you download.


## System requirement
OS: Ubuntu 16.04

Memory: at least 64GB

GPU: Nvidia 1080ti (11GB memory) is **minimum** requirement, and you need to reduce the number of z slices input to the network, by setting train_max_crop_size to for example [112, 240, 240]; we **recommend** using Nvidia Titan RTX (24GB memory) with the default settings.

## Install dependencies
1. Install libs using pip or conda
```
Python 3.7
pytorch 1.1.0 (a must if you want to use tensorboard to monitor the loss)
cuda == 9.0/10.0
```
```
conda install -c conda-forge opencv 
conda install -c kayarre pynrrd 
conda install -c conda-forge pydicom
conda install -c conda-forge tqdm
```

**Please make sure your working directory is src/**
```
cd src
```

2. Install a custom module for bounding box NMS and overlap calculation.

(Only needed if you want to train the model, **NO** need to run this for testing) to build two custom functions.
```
cd build/box
python setup.py install
```

3. In order to use Tensorboard for visualizing the losses during training, we need to install tensorboard.

```
pip install tb-nightly  # Until 1.14 moves to the release channel
```

## Preprocess (optional)
Use utils/preprocess.py to preprocess the converted data.

If you have downloaded the raw and preprocessed data, please remeber to change config.py, or other places if necessary:

line 36 data_dir to '../data/raw'

line 37 preprocessed_data_dir to '../data/preprocessed'

## Train
Change training configuration and data configuration in config.py, especially the path to your preprocessed data.

You can change network configuration in net/config.py, then run training script:
```
python train.py
```

## Evaluating model performance
Please change the train_config['initial_checkpoint'] in config.py to the checkpoint you want to use for evaluating the model performance on test data sets. Then run: 

`python test.py eval`

You should see the results for each patient, where each row is an OAR and the columns are: OAR name, DSC, DSC standard deviation, 95%HD, 95%HD standard deviation.

## Test
`python test.py test --weight $PATH_TO_WEIGHT --dicom-path $DICOM_PATH --out-dir $OUTPUT_DIR`

$PATH_TO_WEIGHT is the path to best model weight used for prediction, e.g. "weights/1001_400.ckpt" or "weights/model_weights"

(If the --weight option is a directory, then the script will consider all files in this directory as weights and perform prediction using all weight files in this direcotry. Then a majority voting will be performed to merge multiple predictions. This is more robust and more accurate.
    
If the --weight option is a file, then simply the single model prediction will be performed.)
