
# Anomaly Detection on Time Series: An Evaluation of Deep Learning Methods

The goal of this repository is to evaluate multivariate time-series anomaly detection algorithms on a common set of datasets discussed in the paper:

A. Garg, W. Zhang, J. Samaran, R. Savitha and C. -S. Foo, "An Evaluation of Anomaly Detection and Diagnosis in Multivariate Time Series," in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2021.3105827.
https://ieeexplore.ieee.org/document/9525836

Arxiv: https://arxiv.org/abs/2109.11428

## Implemented Algorithms

### Raw Signal
Raw signal passes as is. Base method for benchmarking.

### PCA
PCA is used for lossy reconstruction. 

### Univar Auto-encoder
Channel-wise auto-encoders for each channel. 

### Autoencoder
Hawkins, Simon et al. "Outlier detection using replicator neural networks." DaWaK, 2002.

### LSTM-ED
Malhotra, Pankaj et al. "LSTM-based encoder-decoder for multi-sensor anomaly detection." ICML, 2016.

### TCN-ED
Based on the TCN benchmarked by Bai, Shaojie et al. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." arxiv, 2018

### LSTM VAE
Based on Park, Daehyung, Yuuna Hoshi, and Charles C. Kemp. "A multimodal anomaly detector for robot-assisted feeding using an LSTM-based variational autoencoder." IEEE Robotics and Automation Letters 3.3 (2018): 1544-1551.

### Omni-anomaly
Su, Ya, et al. "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network." Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2019.

### MSCRED
Zhang, Chuxu, et al. "A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data." Proceedings of the AAAI Conference on Artificial Intelligence. 2019.

## Additional Algorithms Evaluated

These algorithms are also evaluated but could not be included in the repo at the moment as the original implementations lack a license.

### DAGMM

Zong, Bo, et al. “Deep autoencoding gaussian mixture model for unsupervised anomaly detection.” ICLR, 2018

### Telemanom (NASA LSTM)
Hundman, Kyle, et al. "Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding." Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018.

### BeatGan

Zhou, Bin, et al. "BeatGAN: Anomalous Rhythm Detection using Adversarially Generated Time Series." IJCAI, 2019.

### OCAN

Zheng, Panpan, et al. “One-class adversarial nets for fraud detection.” AAAI, 2019.

## Data sets and how to get them

### Swat
This data set contains 11 days of continuous operation in a water treatment testbed. 7 days’ worth of data was collected 
under normal operation while 4 days’ worth of data was collected with attack scenarios.  During the data collection, all 
network traffic, sensor and actuator data were collected.
To obtain the data one must make a request by filling out 
[this form](https://docs.google.com/forms/d/e/1FAIpQLSfnbjv7ZnDNmV_5ge7OfUc_O_h5yUnj708TFL8dD3o3Yoj9Fw/viewform)
Download the files "SWaT_Dataset_Normal_v1.xlsx" and "SWaT_Dataset_Attack_v0.xlsx" and put it in the location <root-of-the-project>/data/raw/swat/raw

### Wadi
Similar to Swat this data set consists of 16 days of continuous operation, of which 14 days’ worth of data from a water 
distribution testbed was collected under normal operation and 2 days with attack scenarios.
After submitting the same 
[form](https://docs.google.com/forms/d/e/1FAIpQLSfnbjv7ZnDNmV_5ge7OfUc_O_h5yUnj708TFL8dD3o3Yoj9Fw/viewform) as for swat, 
you will receive a link to access a google drive.
Then both csv tables that can be found in the 02_WADI Dataset_19 Nov 2017 folder must be placed at the following 
location: <root-of-the-project>/data/raw/wadi/raw

### Damadics
This data set contains data from actuators collected over several days at the Lublin Sugar Factory.
Data can be downloaded from this [website](http://diag.mchtr.pw.edu.pl/damadics/), each file contains the data acquired 
during one day and the data from October 29th to November 22, 2001 should be downloaded and placed at the following 
location: <root-of-the-project>/data/raw/damadics/raw

### Skab
The Skoltech Anomaly Benchmark testbed consists of a water circulation system and its control system, along with a data-processing and storage system. Examples of the type of anomalies induced include partial valve closures, connecting shaft imbalance, reduced motor power, cavitation and flow disturbances. Train and test splits are provided by the authors. 
The dataset is available at https://www.kaggle.com/yuriykatser/skoltech-anomaly-benchmark-skab/version/1

### Smap and Msl
Get data

You can get the public datasets (SMAP and MSL) using:

wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip

cd data && wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv

Place the folder smap_msl under <root-of-the-project>/data/raw/

### Smd
SMD (Server Machine Dataset) is the folder here: 
https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset

Place the folder 'ServerMachineDataset' at the location <root-of-the-project>/data/raw/

## Usage
Put all the datasets in the data/raw/ directory. 

```bash
git clone https://github.com/astha-chem/mvts-ano-eval.git
conda create -n mvtsenv python=3.6
source activate mvtsenv
python3 -m pip3 install --user --upgrade pip
pip install -r requirements.txt --user
# based on your cuda version or use the cpu only version
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch 
# cpu only
# conda install pytorch==1.2.0 torchvision==0.4.0 cpuonly -c pytorch
python3 setup.py install
get_datasets.sh
python3 main.py
python3 metrics_expts.py
```


## Authors/Contributors
* [Astha Garg](https://github.com/astha-chem)
* [Jules Samaran](https://github.com/jules-samaran)
* [Wenyu Zhang](https://github.com/zwenyu)


## Credits
This repository is forked from https://github.com/KDD-OpenSource/DeepADoTS
[Base implementation for AutoEncoder, LSTM-ED](https://github.com/KDD-OpenSource/DeepADoTS)
[Base implementation for Omni-anomaly](https://github.com/NetManAIOps/OmniAnomaly)  
[Base implementation for VAE-LSTM](https://github.com/TimyadNyda/Variational-Lstm-Autoencoder)
[Base implementation for MSCRED 1](https://github.com/Zhang-Zhi-Jie/Pytorch-MSCRED) [Base implementation for MSCRED 2](https://github.com/SKvtun/MSCRED-Pytorch)
[TCN module](https://github.com/locuslab/TCN)

## Note
The following algorithms have also been evaluated but could not be included in the repo at the moment as the original implementations lack a license:
- DAGMM [original repo](https://github.com/danieltan07/dagmm)
- BeatGAN [original repo](https://github.com/Vniex/BeatGAN)
- OCAN [original repo](https://github.com/PanpanZheng/OCAN)
- NASA LSTM [original repo](https://github.com/khundman/telemanom) (Apache 2.0 license which sets restrictions on sharing)
