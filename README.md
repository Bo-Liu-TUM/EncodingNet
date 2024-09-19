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
|File|Dataset|Model|
| ---- | ---- | ---- |
|`Code_2_finetune_cifar10.py`|CIFAR-10|ResNet-18|
|`Code_3_finetune_cifar100.py`|CIFAR-100|ResNet-20|
|`Code_4_finetune_imagenet.py`|ImageNet|ResNet-50|

<!--
## Verilog code
An example Verilog code of one column in 64×64 MAC array is shown in `/verilog/PE_colunm_Bo_64.v`. 
### Systolic array, exact multiplier, exact adder

![trad-exact-mul-exact-add](/verilog/trad/exact-mul-exact-add.svg)

<div id="mm" class="msgbox"><pre><span class="msg_none">Running Icarus Verilog simulator...</span>
<span class="msg_none">VCD info: dumping is suppressed.</span>
<span class="msg_none">a_left * w_in + sum_in =    1 *    1 +      2 =      3, sum_out =      3 (00003 at 15 ps)</span>
<span class="msg_none">a_left * w_in + sum_in =   -2 *    1 +     -3 =     -5, sum_out =     -5 (1fffb at 25 ps)</span>
<span class="msg_none">a_left * w_in + sum_in =    3 *    1 +     -4 =     -1, sum_out =     -1 (1ffff at 35 ps)</span>
<span class="msg_none">a_left * w_in + sum_in =   -4 *    1 +      5 =      1, sum_out =      1 (00001 at 45 ps)</span>
<span class="msg_none">a_left * w_in + sum_in =    1 *   -1 +      2 =      1, sum_out =      1 (00001 at 65 ps)</span>
<span class="msg_none">a_left * w_in + sum_in =   -2 *   -1 +     -3 =     -1, sum_out =     -1 (1ffff at 75 ps)</span>
<span class="msg_none">a_left * w_in + sum_in =    3 *   -1 +     -4 =     -7, sum_out =     -7 (1fff9 at 85 ps)</span>
<span class="msg_none">a_left * w_in + sum_in =   -4 *   -1 +      5 =      9, sum_out =      9 (00009 at 95 ps)</span>
<span class="msg_hint">Hint: Total mismatched samples is 0 out of 0 samples</span>
<span class="msg_none"></span>
<span class="msg_none">Simulation finished at 100 ps</span>
<span class="msg_none">Mismatches: 0 in 0 samples</span>
<span class="msg_none"></span></pre></div>
-->

## Citation
Waiting for the paper to be accepted.
