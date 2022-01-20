# ProgressNerf

This codebase is intended to be a central point-of-contact for various NeRF and NeRF adjacent techniques for experimentation by themselves, as well as for robotics experimentation within the Laboratory for Progress.

Currently, this codebase only supports the basic NeRF model as described by Mildenhall et al. (2020). Additional architectures are on the todo list, including [Plenoxels](https://alexyu.net/plenoxels/?s=09) and voxelized approaches akin to [NSVF](https://arxiv.org/abs/2007.11571).

## Installation

To install, first clone the repository to some location INSTALL_DIR

```bash
cd $INSTALL_DIR
git clone https://github.com/stanlew7531/ProgressNerf
```

Next, we create a new conda environment and install thost requirements

```bash
conda create --name ProgressNerf python=3.8
conda activate ProgressNerf
conda install --file requirements.txt
```

Finally, we need to compile and install the ProgressNerf code itself. Note that any changes to the Cuda or C++ extensions will require rerunning this command.

```bash
pip install -e ./
```

### Pip setup

To install without conda, first, make sure Python 3.8 is installed:
```bash
sudo apt install python3.8
sudo apt install python3.8-dev
```
Then, make a Python 3.8 virtual environment with your preferred method, for example, `python3.8 -m venv [PATH/TO/ENV]`. Install the dependencies with:
```bash
pip intall -r pip-requirements.txt
```

## Training a model

### Download the dataset

Currently, this codebase only has a dataloader for the Progress Tools dataset as proposed by Pavlasek et al. (2020). This dataset can be downloaded from [here](https://drive.google.com/file/d/1QEexEtgMtlqkY1HtGKCgo_kbZ3DF6C1D/view?usp=sharing).

### Prepare the configuration file

Training a new model requires specifying a configuration yml file. Some examples of these files are in the ./ProgressNerf/configs/OGNerfArch directory. In general, the `baseDir` entry controls where the training process will drop the MLP weights and tensorboard outputs. If the training process needs to be restarted, the `load_from_epoch` entry will specify which epoch's set of weights from the base directory to load and resume training from. If the `load_from_epoch` is a negative number, the training process will grab the highest epoch value in the previous training process's output. If the `load_from_epoch` value is not specified, or is None, then the training process will not load any values and will begin training fresh from epoch 0.

For the train and test dataloaders, it is important that the `baseDataDir` value be changed to reflect the unzipped location of the ToolsParts dataset. Consult that dataset to determine what scenes the training process should use. The `samplesLimit` value controls how many images to include in the dataloader - specifying a value of -1 results in all samples being loaded.

### Begin the training process

To start the training process with a user provided yml named `<some_config_file.yml>`, run:

```bash
python ProgressNerf/Architectures/OGNerfArch.py <some_config_file.yml>
```

Note that if no user yaml file is provided, ProgressNerf will default to the model as specified in the ./ProgressNerf/configs/OGNerfArch/toolPartsCoarseFinePerturbed.yml file.

This will start the training process and display the per-epoch progress bar. In order to monitor the overall training process and view the evaluation outputs, in a new terminal, navigate to the base directory specified in the configuration yaml, and execute:

```bash
conda activate ProgressNerf
tensorboard --logdir ./
```

In a web browser, you should then be able to navigate to [http://localhost:6006](http://localhost:6006) to see train & test error metrics, along with the images produced by the evaluation steps next to their ground truths.

## Rendering from arbitrary camera poses

If a full training config is specified, calling the `doEvalRendering` function in the architecture object will allow for rendering at user supplied camera poses. If this task needs to be accomplished without being in a training environment, the YAML configuration should not specify a training dataloader. The MLP weights should still be loaded under such a scenario, but neither the optimizer nor any other training specific field will be. This will allow for `doEvalRendering` to be called and RGB images returned. For an example, the tooPartsCoarseFinePerturbedEvalOnly.yml config contains the minimal amount of information necessary to perform evaluation renderings, and the Sandbox.ipynb notebook contains an example of how to setup and render images from a pre-trained model. 
