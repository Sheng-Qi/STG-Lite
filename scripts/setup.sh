#! /bin/bash

conda create -n stg-lite python=3.12
conda activate stg-lite

conda install cudatoolkit=11.8

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install pyyaml tqdm plyfile kornia opencv-python pydantic

pip install submodules/diff-gaussian-rasterization/
pip3 install submodules/forward_lite
pip3 install submodules/gaussian_rasterization_ch3
pip3 install submodules/simple-knn/
