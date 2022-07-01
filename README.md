# Variational Causal Inference

![](figure/elbo.png)


## Installation

### 1. Create Conda Environment
```bash
conda config --append channels conda-forge
conda create -n vci-env --file requirements.txt
conda activate vci-env
```

### 2. Install Learning Libraries
- [Pytorch](https://pytorch.org/) [**1.11**.0](https://pytorch.org/get-started/previous-versions/)

  \* *make sure to install the right versions for your toolkit*


## Run
Once the environment is set up, the function call to train the model is:

```bash
./main_train.sh &
```

A list of flags may be found in `main_train.sh` and `main_train.py` for experimentation with different network parameters. The run log and models are saved under `*artifact_path*/saves`, and the tensorboard log is saved under `*artifact_path*/runs`.


## License

Contributions are welcome! All content here is licensed under the MIT license.
