# Learning Using Privileged Information for Zero-Shot Action Recognition
This is an official implementation of:

Zhiyi Gao, et al. Learning Using Privileged Information for Zero-Shot Action Recognition, ACCV, 2022. [Arxiv Version](https://arxiv.org/abs/2206.08632)

**Note**: This is a preliminary release, but we feel that it would be better to first put the code out here.
# Install
## Requirements

Run install.sh to get the uncommon libraries (faiss, tensorboardx, joblib) and the latest version of pytorch compatible with cuda 9.2.
## Getting datasets

OlympicSports can be downloaded [here](http://vision.stanford.edu/Datasets/OlympicSports/)

HMDB51 can be downloaded [here](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

UCF101 can be downloaded [here](https://www.crcv.ucf.edu/research/data-sets/ucf101/)

## Grt R(2+1)D Pretrained Model

The R(2+1)D models, used in the paper, can be downloaded [here](https://drive.google.com/drive/folders/1mXybSSZk5FtLzf5Vc5KX6WbhtPeXXzeI?usp=sharing)

## Get the word2vec model

```
sudo chmod + assets/download_word2vec.sh
./assets/download_word2vec.sh
```
or can be downloaded [here](https://drive.google.com/drive/folders/1YIm6zIMBU7dP40nDSimurZVXNpfcRrUw?usp=sharing)
## Get BiT Model

The BiT models, used in the paper, can be downloaded [here](https://drive.google.com/drive/folders/1u50fVwWnT-fAg983TGXJ2XxEfmc9QRPE?usp=sharing)

# Training


# Citation
If you find this repository useful, please cite our paper:

```
@article{gao2022learning,
  title={Learning Using Privileged Information for Zero-Shot Action Recognition},
  author={Gao, Zhiyi Hou and Hou, Yonghong and Li, Wanqing and Guo, Zihui and Yu, Bin},
  journal={arXiv preprint arXiv:2206.08632},
  year={2022}
}
```
# Acknowledgement
- [E2E](https://github.com/bbrattoli/ZeroShotVideoClassification)
- [BiT](https://github.com/google-research/big_transfer)
