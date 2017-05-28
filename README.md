# GEDI CNN project

## If you're using this project you will want to do one of four things (advanced operations, like ratio and time-course prediction will be added below):
1. Convert data into a CNN-friendly format.
2. Train a model.
3. Test data on a model.
4. Visualize why a model made the decisions it made.
## I will walk you through each of these steps. But first, some basic familiarization with the project.

* `gedi_config.py` contains all of the settings for the project.
  * You will need to set `self.home_dir` and `self.src_dir` to folders on your machine.
  * You will need to put your GEDI TIFF images into the directory pointed to by `self.original_image_dir`.
  * Set `self.which_dataset` to the image panel you're testing -- GFP/masked-GFP/GEDI/ratio.
  * Set `self.experiment_image_set` to match whatever the structure of the image folders you are using. It is expected that folders take the format `{XXX}_{YYY}` where {XXX} is the image category (e.g. Live or Dead) and {YYY} is the stem indicated by `self.experiment_image_set`. An example is `Live_rat_gedi_images`.
  * Take a look at the other settings, but if you're using a typical set-up is unlikely you'll have to change them.



## 1. Convert data into a CNN-friendly format.
To combat bottlenecks in data transfer related to python and CPU memory swapping, we use a ``memory-mapped'' formate with our CNNs. Since this is tensorflow, we use ``tf-records''. The following will walk you through how to create these.

* Option A: I want to convert data into a tf-records to use on a model that's already been trained.
  * Edit line 34 in `gedi_config.py`. Set `self.easy_analysis = True`. Now whatever image folders you point the script to in `self.experiment_image_set` will be captured into a single tf-records file, that can then be passed through a testing script.

* Option B: I want to convert data into a tf-records for training/validation (no seperate testing data).
  * Edit line 34 in `gedi_config.py`. Set `self.easy_analysis = False`. Set `test_set = False`. 

* Option C: I want to convert data into a tf-records for training/validation with a seperate testing set.
  * Edit line 34 in `gedi_config.py`. Set `self.easy_analysis = False`. Set `test_set = True`.

* Option D: I want to package a ``blinded'' dataset, without its class labels.
  * Edit line 34 in `gedi_config.py`. Set `self.easy_analysis = False`. Set `blinded = True`.

## After choosing your appropriate settings, run `sh run_preprocessing_scipts.sh`

## 2. Train a model.
After preparing data in a CNN-friendly format, you want to train a model. Look in `gedi_config.py` for settings that you can adjust for CNN training. 

## Run `sh train_models.sh X` where X is the GPU you wish to run training on. 

## 3. Test data on a model.
* Option A: Test a model on any dataset (assuming similar charactaristics of both, e.g. both are Rat neurons). (model_dir is your model, validation_data is the tf-records file you want to test on.)

```python training_and_eval/test_vgg16.py --model_dir=/media/data/GEDI/drew_images/p
roject_files/train_checkpoint/gfp_2017_05_27_13_56_55 --validation_data=/media/data/GEDI/drew_images/project_files/tfrecords/all_rh_analysis_rat_gfp/test.tfrecords --selected_ckpts=32```

* Option B: Train an SVM on a model to optimize the transfer of its predictions to a new dataset, e.g. trained on rat and tested on human.

```python training_and_eval/test_vgg16_tf_svm.py --model_dir=/media/data/GEDI/drew_images/p
roject_files/train_checkpoint/gfp_2017_05_27_13_56_55 --validation_data=/media/data/GEDI/drew_images/project_files/tfrecords/all_rh_analysis_rat_gfp/test.tfrecords --selected_ckpts=32```

## 4. Visualize model decisions in pixel space -- why did the model make the decisions it made?
* Option A: Visualize decisions on any dataset. (model_dir is your model, validation_data is the tf-records file you want to test on.)

```python visualization/gedi_lrp_tfrecords.py --model_dir=/media/data/GEDI/drew_images/project_files/train_checkpoint/gfp_2017_05_27_13_56_55 --validation_data=/media/data/GEDI/drew_images/project_files/tfrecords/all_rh_analysis_rat_gfp/test.tfrecords --selected_ckpts=32```
 

