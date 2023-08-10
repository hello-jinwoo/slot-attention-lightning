<div align="center">

# Manipulating Object-Centric Representation with User's Intention \\ (NeurIPS2023 under review)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539) -->
<!-- [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

## Description

Implementation code for the paper under review NeurIPS2023.

<br>

## Installation

```bash
# clone project
git clone https://github.com/janghyuk-choi/slotaug
cd slotaug

# [OPTIONAL] create conda environment
conda create -n slotaug python=3.9
conda activate slotaug

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

<br>

## Before running command...
> ```bash
> # when you use s31 with gpu0, set env variables as follows:
> echo 'export SERVER_NAME="server_name"' >> ~/.bashrc
> echo 'export GPU_NAME="gpu_num"' >> ~/.bashrc
> source ~/.bashrc
> echo $SERVER_NAME
> echo $GPU_NAME
> ```  

<br>

## Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/).

```bash
# v1
HYDRA_FULL_ERROR=1 \
CUDA_VISIBLE_DEVICES=0 \
python src/train.py \
experiment=slotaug/clv6.yaml \
data.data_dir={data_dir} \
data.name="clv6" \
data.transform_contents=\'scale,translate,color\' \
+model.loss_sc_weight=0.0 \
+model.net.aux_identity=False \
model.name="v1" 

# v2
HYDRA_FULL_ERROR=1 \
CUDA_VISIBLE_DEVICES=0 \
python src/train.py \
experiment=slotaug/clv6.yaml \
data.data_dir={data_dir} \
data.name="clv6" \
data.transform_contents=\'scale,translate,color\' \
+model.loss_sc_weight=0.0 \
+model.net.aux_identity=True \
model.name="v2" 

# v3
HYDRA_FULL_ERROR=1 \
CUDA_VISIBLE_DEVICES=0 \
python src/train.py \
experiment=slotaug/clv6.yaml \
data.data_dir={data_dir} \
data.name="clv6" \
data.transform_contents=\'scale,translate,color\' \
+model.loss_sc_weight=0.1 \
+model.net.aux_identity=True \
model.name="v3" 
