# Fast Graph Generation via Spectral Diffusion

Code for the paper [Fast Graph Generation via Spectral Diffusion](https://ieeexplore.ieee.org/abstract/document/10366850) (IEEE TPAMI 2023).

## Dependencies

GSDM is built in **Python 3.8.0** and **Pytorch 1.10.1**. Please use the following command to install the requirements:

however, molsets can only be directly downloaded via pip while in python 3.7

```sh
pip install -r requirements.txt
```

eden包下载：
来源：[eden](https://github.com/fabriziocosta/EDeN)

```sh
pip install git+https://github.com/fabriziocosta/EDeN.git --user
```

```sh
git clone https://github.com/fabriziocosta/EDeN.git
cd EDeN
pip install -e .
```

moses包下载


## Running Experiments



### Train model

```sh
CUDA_VISIBLE_DEVICES=${gpu_ids} python main.py --type train --config ${train_config}
```

for example, 

```sh
CUDA_VISIBLE_DEVICES=0 python main.py --type train --config community_small
python main.py --type train --config Digg
```

### Evaluation

For the evaluation of generic graph generation tasks, run the following command to compile the ORCA program (see http://www.biolab.si/supp/orca/orca.html):

```sh
cd evaluation/orca 
g++ -O2 -std=c++11 -o orca orca.cpp
```

### Generate

To generate graphs using the trained score models, run the following command.

```sh
CUDA_VISIBLE_DEVICES=${gpu_ids} python main.py --type sample --config community_small
```


## Citation

If you found the provided code with our paper useful in your work, we kindly request that you cite our work.

```BibTex
@article{luo2023fast,
  title={Fast graph generation via spectral diffusion},
  author={Luo, Tianze and Mo, Zhanfeng and Pan, Sinno Jialin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
  url={https://ieeexplore.ieee.org/abstract/document/10366850}
}
```