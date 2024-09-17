# EncodingNet
Open source for a submitted paper on IEEE Transactions on Circuits and Systems for Artificial Intelligence.

## Pretrained models
|  Dataset   | Online Models  |
|  ----  | ----  |
| CIFAR-10  | [https://github.com/huyvnphan/PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10) |
| CIFAR-100  | [https://github.com/weiaicunzai/pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100) |
| ImageNet | [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)    |

## CGP (Cartesian Genetic Programming) package
`cgp` is a modified package which is copied from [https://zenodo.org/records/3889163](https://zenodo.org/records/3889163)

## User-defined encoding package
`encode_tools` is a user-defined package which includes functions related to encoding.

## User-defined model package
`models` is a user-defined package which implement encoding techniques on modified `Linear` and `Conv2d` layers.

## CGP search
`Code_1_CGP_search.py` is used to apply CGP search process.

## Fine-tune neural networks
The following files are used to apply searched encodings, fine-tune encoding-based neural networks, and test for accuracies. 
`Code_2_finetune_cifar10.py`, `Code_3_finetune_cifar100.py` and `Code_4_finetune_imagenet.py`.


