<h1 align="center">
<p>Big Data Project (CSGY- 6513) :bar_chart:</p>
<p align="center">
<img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg">
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.8-blue?logo=python">
<img alt="tensorflow" src="https://img.shields.io/badge/tensorflow-%3D2.8.0-orange?logo=tensorflow">
<img alt="PyPI" src="https://img.shields.io/badge/release-v1.0-brightgreen?logo=apache&logoColor=brightgreen">
</p>
</h1>

<h2 align="center">
<p>Image Enhancement using Distributed Deep Learning</p>
</h2>

## Supervised by Prof. Juan Rodriguez 

### Built by 
- Niharika Krishnan (nk2982)
- Vaibhav Singh (vs2410)


## Table of Contents

<!-- TOC -->

- [Dataset](#dataset)
- [Installation](#installation)  


<!-- /TOC -->

### Installation

```bash
git clone https://github.com/Distributed_Image_Enhancement.git
cd Distributed_Image_Enhancement
```
1.) Installation on conda environment -  
```bash
conda env create --name v_env --file=environments.yml
python3 train.py
```
2.) Installation via requirements.txt -
```bash
pip install requirements.txt
python3 train.py
```
  
### Dataset
https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq
(21gb)

1. Currently the program uses 10 images for running end to end.
2. Download the complete dataset from the above url into "face_data" folder and change the data_path in config.yml file

    
