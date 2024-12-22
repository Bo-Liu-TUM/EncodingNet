# EncodingNet
This ia an open-source repo for an accepted paper, `EncodingNet: A Novel Encoding-based MAC Design for Efficient Neural Network Acceleration`, on `IEEE Transactions on Circuits and Systems for Artificial Intelligence (TCASAI)`. 

[//]: # (The codes in this repo will be well-structured and updated in the following weeks.)

## Pretrained Models
| Dataset   | Online Models                                                       | Packages               |
|-----------|---------------------------------------------------------------------|------------------------|
| CIFAR-10  | [PyTorch CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10)     | `./cifar10_models`     |
| CIFAR-100 | [PyTorch CIFAR100](https://github.com/weiaicunzai/pytorch-cifar100) | `./cifar-models`       |
| ImageNet  | [TorchVision](https://pytorch.org/vision/stable/models.html)        | `torchvision.models`   |

## CGP (Cartesian Genetic Programming) Package
`./cgp` is a modified package which is copied from [https://zenodo.org/records/3889163](https://zenodo.org/records/3889163)

## User-defined Packages
`./encode_tools` is a user-defined package which includes functions related to encoding.

`./models` is a user-defined package which implements encoding techniques on modified `Linear` and `Conv2d` layers.

## CGP Search
`./Code_1_CGP_search.py` is used to apply CGP search process.

### Command 

| Command             | Default   | Choices                                                     | Description                              |
|---------------------|-----------|-------------------------------------------------------------|------------------------------------------|
| `--gpu`             | `0`       | `0`, `1`, `2`, `3`                                          | gpu device index                         |
| `--target`          | `64`      | `36`, `40`, `42`, `44`, `46`, `48`, `52`, `56`, `60`, `64`  | the target product bit-width             |
| `--search`          | `128`     | `64`, `128`, `256`                                          | total output nodes                       |
| `--cols`            | `2`       | `1`, `2`, `3`                                               | columns of node array                    |
| `--rows`            | `256`     | `64`, `128`, `256`                                          | rows of node array                       |
| `--th`              | `0.1`     | `0.1`, `0.2`, `0.5`, `1`, `1.5`, `2`, `5`, `10`, `20`       | threshold of maximal relative error      |
| `--gen`             | `2500`    | integer                                                     | generations                              |
| `--idx`             | `0`       | `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`            | the index of the search                  |
| `--n-parents`       | `10`      | `10`, `20`, `30`, `40`, `50`                                | parents maintained                       |
| `--n-offsprings`    | `50`      | `10`, `20`, `30`, `40`, `50`                                | offsprings generated                     |
| `--n-champions`     | `2`       | `1`, `2`, `3`, `4`, `5`                                     | champions saved                          |
| `--mutate-strategy` | `dynamic` | `dynamic`, `fixed`                                          | strategy to change mutation rate         |
| `--mutate-rate`     | `0.1`     | `0.01`, `0.025`, `0.05`, `0.10`, `0.15`, `0.20`             | only valid if --mutate-strategy is fixed |

### Usage

```commandline
python Code_1_CGP_search.py --gpu 0 --target 64 --search 128 --cols 2 --rows 256 --th 0.1 --gen 2500 --idx 0 --n-parents 10 --n-offsprings 50 --n-champions 2 --mutate-strategy dynamic --mutate-rate 0.1 
```

## Test and Fine-tune Neural Networks
The file `./Code_2_finetune_models.py` is used to apply searched encodings, fine-tune encoding-based neural networks, and test for accuracies.

### Command
| Command           | Default            | Choices                                                                                    | Description               |
|-------------------|--------------------|--------------------------------------------------------------------------------------------|---------------------------|
| `--arch`          | `resnet18`         | `resnet18`, `mobilenet_v2`, `resnet20`, `mobilenetv2_x0_5`, `resnet50`, `efficientnet_b0`  | model name                |
| `--data`          | `cifar10`          | `cifar10`, `cifar100`, `imagenet2012`                                                      | dataset                   |
| `--run`           | `test`             | `retrain`, `test`                                                                          | running mode              |
| `--epochs`        | `25`               | integer                                                                                    | epochs                    |
| `--batch-size`    | `256`              | integer                                                                                    | batch size                |
| `--gpu`           | `0`                | `0`, `1`, `2`, `3`                                                                         | gpu devices index         |
| `--workers`       | `4`                | integer                                                                                    | workers to load dataset   |
| `--print-freq`    | `1`                | integer                                                                                    | print frequency           |
| `--running-cache` | `./running_cache/` | directory path                                                                             | path save searched result |
| `--mode`          | `FP32`             | `FP32`, `Exact-INT`, `Approx-INT`                                                          | data representation mode  |
| `--a-bit`         | `8`                | `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `32`                                               | bit-width of activation   |
| `--w-bit`         | `8`                | `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `32`                                               | bit-width of weights      |
| `--product-bit`   | `0`                | `36`, `40`, `42`, `44`, `46`, `48`, `52`, `56`, `60`, `64`                                 | bit-width of products     |

`--run` has two options. `test` is for inference and `retrain` for fine-tune.

`--mode` has three options. `FP32` is for 32-bit floating point, `Exact-INT` for 1~8-bit exact integer multiplication, and `Approx-INT` for 8-bit approximate integer multiplication.


### Test Mode

#### FP32

```commandline
python Code_2_finetune_models.py --data cifar10 --arch resnet18 --run test --mode FP32 
```

#### Exact-INT

```commandline
python Code_2_finetune_models.py --data cifar10 --arch resnet18 --run test --mode Exact-INT --a-bit 8 --w-bit 8
```

#### Approx-INT

```commandline
python Code_2_finetune_models.py --data cifar10 --arch resnet18 --run test --mode Approx-INT --a-bit 8 --w-bit 8 --product-bit 64
```

### Fine-tune Mode

| Dataset   | Models          | Learning Rate | Batch Size |
|-----------|-----------------|---------------|------------|
| CIFAR-10  | ResNet-18       | 1e-4          | 256        |
| CIFAR-10  | MobileNet-V2    | 1e-4          | 256        |
| CIFAR-100 | ResNet-20       | 1e-4          | 256        |
| CIFAR-100 | MobileNet-V2    | 1e-3          | 256        |
| ImageNet  | ResNet-50       | 1e-5          | 32         |
| ImageNet  | EfficientNet-B0 | 1e-5          | 32         |

#### FP32

```commandline
python Code_2_finetune_models.py --data cifar10 --arch resnet18 --run retrain --mode FP32 
```

#### Exact-INT

```commandline
python Code_2_finetune_models.py --data cifar10 --arch resnet18 --run retrain --mode Exact-INT --a-bit 8 --w-bit 8
```

#### Approx-INT

```commandline
python Code_2_finetune_models.py --data cifar10 --arch resnet18 --run retrain --mode Approx-INT --a-bit 8 --w-bit 8 --product-bit 64
```


## Citation

The paper is now accepted, and can be accessed online, [https://ieeexplore.ieee.org/document/10746865](https://ieeexplore.ieee.org/document/10746865).


### Plain Text

Bo Liu, Bing Li, Grace Li Zhang, Xunzhao Yin, Cheng Zhuo and Ulf Schlichtmann, "EncodingNet: A Novel Encoding-based MAC Design for Efficient Neural Network Acceleration," _IEEE Transactions on Circuits and Systems for Artificial Intelligence (TCASAI)_, vol. 1, no. 2, pp. 164-177, 2024.


### BibTeX
```
@ARTICLE{10746865,
  author={Liu, Bo and Li, Bing and Zhang, Grace Li and Yin, Xunzhao and Zhuo, Cheng and Schlichtmann, Ulf},
  journal={IEEE Transactions on Circuits and Systems for Artificial Intelligence (TCASAI)}, 
  title={{EncodingNet}: A Novel Encoding-based {MAC} Design for Efficient Neural Network Acceleration}, 
  year={2024},
  volume={1},
  number={2},
  pages={164-177},
  doi={https://doi.org/10.1109/TCASAI.2024.3493035},
  url={https://ieeexplore.ieee.org/document/10746865}
}
```
