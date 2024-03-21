# MESM
The official code of [Towards Balanced Alignment: Modal-Enhanced Semantic Modeling for Video Moment Retrieval](https://arxiv.org/abs/2312.12155) (AAAI 2024)

## Introduction
MESM focuses on the modality imbalance problem in VMR, which means the semantic richness inherent in a video far exceeds that of a given limited-length sentence. The problem exists at both the frame-word level and the segment-sentence level.

<div align="center">
    <img src="images/motivation.png" width="500px">
</div>

MESM proposes the modal-enhanced semantic modeling for both levels to address this problem.
![Pipeline](images/pipeline.png)

## Prerequisites
This work was tested with Python 3.8.12, CUDA 11.3, and Ubuntu 18.04. You can use the provided docker environment or install the environment manully.

### Docker
Assuming you are now at path `/`.
```bash
git clone https://github.com/lntzm/MESM.git
docker pull lntzm/pytorch1.11.0-cuda11.3-cudnn8-devel:v1.0
docker run -it --gpus=all --shm-size=64g --init -v /MESM/:/MESM/ lntzm/pytorch1.11.0-cuda11.3-cudnn8-devel:v1.0 /bin/bash
# You should also download nltk_data in the container.
python -c "import nltk; nltk.download('all')"
```

### Conda Environment
```bash
conda create -n MESM python=3.8
conda activate MESM
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
# You should also download nltk_data.
python -c "import nltk; nltk.download('all')"
```

## Data Preparation
The structure of the data folder is as follows:
```bash
data
├── charades
│   ├── annotations
│   │   ├── charades_sta_test.txt
│   │   ├── charades_sta_train.txt
│   │   ├── Charades_v1_test.csv
│   │   ├── Charades_v1_train.csv
│   │   ├── CLIP_tokenized_count.txt
│   │   ├── GloVe_tokenized_count.txt
│   │   └── glove.pkl
│   ├── clip_image.hdf5
│   ├── i3d.hdf5
│   ├── slowfast.hdf5
│   └── vgg.hdf5
├── Charades-CD
│   ├── charades_test_iid.json
│   ├── charades_test_ood.json
│   ├── charades_train.json
│   ├── charades_val.json
│   ├── CLIP_tokenized_count.txt -> ../charades/annotations/CLIP_tokenized_count.txt
│   └── glove.pkl -> ../charades/annotations/glove.pkl
├── Charades-CG
│   ├── novel_composition.json
│   ├── novel_word.json
│   ├── test_trivial.json
│   ├── train.json
│   ├── CLIP_tokenized_count.txt -> ../charades/annotations/CLIP_tokenized_count.txt
│   └── glove.pkl -> ../charades/annotations/glove.pkl
├── qvhighlights
│   ├── annotations
│   │   ├── CLIP_tokenized_count.txt
│   │   ├── highlight_test_release.jsonl
│   │   ├── highlight_train_release.jsonl
│   │   ├── highlight_val_object.jsonl
│   │   └── highlight_val_release.jsonl
│   ├── clip_image.hdf5
│   └── slowfast.hdf5
├── TACoS
│   ├── annotations
│   │   ├── CLIP_tokenized_count.txt
│   │   ├── GloVe_tokenized_count.txt
│   │   ├── test.json
│   │   ├── train.json
│   │   └── val.json
│   └── c3d.hdf5
```
All extracted features are converted to `hdf5` files for better storage. You can use the provided python script `./data/npy2hdf5.py` to convert `*.npy` or `*.npz` files to an `hdf5` file.

### CLIP_tokenized_count.txt & GloVe_tokenized_count.txt
These files are built for masked language modeling in FW-MESM, and they can be generated by running
```bash
python -m data.tokenized_count
```

- `CLIP_tokenized_count.txt`
    
    Column 1 is the word_id tokenized by the CLIP tokenizer, column 2 is the times the word_id appears in the whole dataset.

- `GloVe_tokenized_count.txt`
    
    Column 1 is the splited word in a sentence, column 2 is its tokenized id for GloVe, and column 3 is the times the word appears in the whole dataset.


### Charades Features
We provide the merged `hdf5` files of *CLIP* and *SlowFast* features [here](https://mailustceducn-my.sharepoint.com/:f:/g/personal/liuzhihang_mail_ustc_edu_cn/EuzuWDwYPX1Kvceb9px5Y1YBHmSfL_1ptk8CVuInCyDBdg?e=e14Qam). However, *VGG* and *I3D* features are too large for our network drive storge space. In fact, we just followed [QD-DETR](https://github.com/wjun0830/QD-DETR) to get video features for all extractors. They provide detailed ways to obtain features, see this [link](https://github.com/wjun0830/QD-DETR/issues/1#issuecomment-1493414922).

`glove.pkl` records the necessary vocabulary for the dataset. Specifically, it contains the most common words for MLM, the wtoi dictionary, and the id2vec dictionary. We use the `glove.pkl` from [CPL](https://github.com/minghangz/cpl/blob/main/data/charades/glove.pkl), which can also be built from the standard `glove.6B.300d`.


### QVHighlights Features
Same as [QD-DETR](https://github.com/wjun0830/QD-DETR), we also use the official feature files for QVHighlights dataset from [Moment-DETR](https://github.com/jayleicn/moment_detr), which can be downloaded [here](https://drive.google.com/file/d/1Hiln02F1NEpoW8-iPZurRyi-47-W2_B9/view?usp=sharing), and merge them to `clip_image.hdf5` and `slowfast.hdf5`.


### TACoS Features
We follow [VSLNet](https://github.com/26hzhang/VSLNet) to get the C3D features for TACoS. Specifically, we run `prepare/extract_tacos_org.py` and set the `sample_rate` 128 to extract the pretrained C3D visual features from [TALL](https://github.com/jiyanggao/TALL) and then convert it to `hdf5` file. We provide the converted file [here](https://mailustceducn-my.sharepoint.com/:f:/g/personal/liuzhihang_mail_ustc_edu_cn/EuzuWDwYPX1Kvceb9px5Y1YBHmSfL_1ptk8CVuInCyDBdg?e=e14Qam).


## Trained Models
| Dataset | Extractors | Download Link |
| :--: | :--: | :--: |
| Charades-STA | VGG, GloVe | [OneDrive](https://mailustceducn-my.sharepoint.com/:f:/g/personal/liuzhihang_mail_ustc_edu_cn/EmdVX_iFFhZDqZbs6w2trcUBuSvV7kFdrPZEMNwYsVX0Wg?e=B93ns1) |
| Charades-STA | C+SF, C | [OneDrive](https://mailustceducn-my.sharepoint.com/:f:/g/personal/liuzhihang_mail_ustc_edu_cn/EnpsfEa7bl5DoLINN0vlHYwBf_pNBSL1-uc5Mm34NwioYg?e=zqwtZD) |
| Charades-CG | C+SF, C | [OneDrive](https://mailustceducn-my.sharepoint.com/:f:/g/personal/liuzhihang_mail_ustc_edu_cn/EsGUsf-DfU5NoNJxC8K5iqwBMn1uUxt2WYuNIheKTXQOTw?e=ot50zt) |
| TACoS | C3D, GloVe | [OneDrive](https://mailustceducn-my.sharepoint.com/:f:/g/personal/liuzhihang_mail_ustc_edu_cn/EpWhaB9jPANJjzX-s0KMrQ8BICETYM0D2soMdNFphEHdAw?e=pFFDY4) |
| QVHighlights | C+SF, C | [OneDrive](https://mailustceducn-my.sharepoint.com/:f:/g/personal/liuzhihang_mail_ustc_edu_cn/EgBW1_p2_ZFLn3KeGpbfqAkBDbKlaO-njd7AHyGCc52zMQ?e=L5ftaU) |


## Training

You can run `train.py` with args in command lines:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py {--args}
```

Or run with a config file as input:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config_file ./config/charades/VGG_GloVe.json
```


## Evaluation

You can run `eval.py` with args in command lines:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py {--args}
```

Or run with a config file as input:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --config_file ./config/charades/VGG_GloVe_eval.json
```

## Citation
If you find this repository useful, please use the following entry for citation.

```
@article{liu2023towards,
  title={Towards Balanced Alignment: Modal-Enhanced Semantic Modeling for Video Moment Retrieval},
  author={Liu, Zhihang and Li, Jun and Xie, Hongtao and Li, Pandeng and Ge, Jiannan and Liu, Sun-Ao and Jin, Guoqing},
  journal={arXiv preprint arXiv:2312.12155},
  year={2023}
}
```

## Acknowledgements
This implementation is based on these repositories:
- [QD-DETR](https://github.com/wjun0830/QD-DETR)
- [CPL](https://github.com/minghangz/cpl)
