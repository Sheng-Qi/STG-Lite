# Spacetime Gaussian Lite (STG-Lite)

## Overview

This is a lightweight version of the Spacetime Gaussian (STG) model. 

This project is based on the original project [Spacetime Gaussian](https://github.com/oppo-us-research/SpacetimeGaussians). The main differences are:

- Integrated hyperparameters into the configuration file
- Simplified the code structure, removed redundant code and modularized the dataset and model
- Only kept the `ours_lite` model and `Technicolor` dataset from the original project
- Removed the guided sampling part

## Installation

```bash
bash scripts/setup.sh
```

## Usage

Modify the configuration file `config/default.yaml` and run the following command:

```bash
python main.py --config config/default.yaml --log INFO
```

## Note

Other Detailed instructions can be found in the original project [Spacetime Gaussian](https://github.com/oppo-us-research/SpacetimeGaussians).

This project is still under development, so there might be some bugs. Feel free to open an issue if you find any.
