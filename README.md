# Adaptive Hypergraph Convolutional Network for No-Reference 360-degree Image Quality Assessment

## TODO
- [x] ~~Code release on the CVIQD dataset~~
- [ ] Release the code for the OIQA dataset

## Introduction
For the implementation details of AHGCN, please see the file "./model/cviqd.py".

## Train and Test
First, download datasets from [Here](https://drive.google.com/file/d/1ihDsUkqL58LRL7yCQC5lCp8EHnCC_9EX/view?usp=drive_link).  

Second, modify the datapath in line 53 in the datasets/cviqd.py:

Thrid, run the following command to train and test the model

```
python train_cviqd_shell.py
```
The best model and its corresponding predictions are saved in the directory "./save/" and the directory "./mat/", respectively.

We have uploaded the checkpoint of the best model retrained on a single NVIDIA RTX 4090 GPU in the directory "./save/best_cviqd"

## Citation
Please cite the following paper if you use this repository in your reseach.
```
@inproceedings{fu2022adaptive,
  title={Adaptive hypergraph convolutional network for no-reference 360-degree image quality assessment},
  author={Fu, Jun and Hou, Chen and Zhou, Wei and Xu, Jiahua and Chen, Zhibo},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={961--969},
  year={2022}
}
```
## Contact
For any questions, feel free to contact: `fujun@mail.ustc.edu.cn`