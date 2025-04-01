# CLFDA: Continuous Low-Frequency Decomposition Architecture for Fine-Grained Land Use Classification

## Requirements
- This code is written for `python3.10`.
- Follow the installation instructions of the VMamba project to install the [RS-Mamba(rsm-ss)](https://github.com/walking-shadow/Official_Remote_Sensing_Mamba)  environment.
- pytorch = 2.1.1
- torchvision
- selective-scan-0.0.2
- pytorch_wavelets
- numpy, prettytable, tqdm, scikit-learn, matplotlib, argparse, h5py, timm, einops
- [https://blog.csdn.net/yyywxk/article/details/136071016](https://blog.csdn.net/yyywxk/article/details/140422758)


## Training and Evaluating
The pipeline for training with CLFDA is the following:

1. "option.py"
- `args.dataset = 'gid15'`
- `args.arch = 'CLFDA'`

2. Train and Evaluate the model. For example, to run an experiment for the GID-15 and FUSU dataset,  run:

- `python main.py`


## Acknowledgment
This code is heavily borrowed from [RS-Mamba(rsm-ss)](https://github.com/walking-shadow/Official_Remote_Sensing_Mamba) and [RS3Mamba](https://github.com/sstary/SSRS/tree/main/RS3Mamba)


## Citation
If you find our work useful in your research, please consider citing our paper:

```

```
## Contact
Please contact houdongyang1986@163.com if you have any question on the codes.
