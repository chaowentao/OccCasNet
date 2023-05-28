# OccCasNet: Occlusion-aware Cascade Cost Volume for Light Field Depth Estimation



<!-- [Paper link](https://arxiv.org/pdf/2208.09688) -->



## Network Architecture
![Network Architecture](images/network.png)

# SOTA on 4D Light Field Benchmark
- Our method achieve competitive performance on the HCI 4D LF Benchmark in terms of all the five accuracy
metrics (i.e., BadPix0.01, BadPix0.03, BadPix0.07, MSE and Q25).

<img src="images/benchmark.png" width="555" align=center />  

- For more detail comparison, please use the link below.
- [Benchmark link](https://lightfield-analysis.uni-konstanz.de/benchmark/table?column-type=images&metric=badpix_0070)

# Environment
```
Ubuntu            16.04
Python            3.8.10
Tensorflow-gpu    2.5.0
CUDA              11.2
```

# Train OccCasNet
1. Download HCI Light field dataset from <http://hci-lightfield.iwr.uni-heidelberg.de/>.  
2. Unzip the LF dataset and move **'additional/, training/, test/, stratified/ '** into the **'hci_dataset/'**.
4. **Stage 1:** Run `python train_occcas.py`
  - Checkpoint files will be saved in **'LF_checkpoints/XXX_ckp/iterXXXX_valmseXXXX_bpXXX.hdf5'**.
  - Training process will be saved in 
    - **'LF_output/XXX_ckp/train_iterXXXXX.jpg'**
    - **'LF_output/XXX_ckp/val_iterXXXXX.jpg'**.

# Evaluate OccCasNet
- Run `python evaluation_occcas.py`
  - `path_weight='LF_checkpoint/SubFocal_sub_0.5_js_0.1_ckp/iter0010_valmse0.768_bp1.93.hdf5'`

# Submit OccCasNet
- Run `python submission_occcas.py`
  - `path_weight='LF_checkpoint/SubFocal_sub_0.5_js_0.1_ckp/iter0010_valmse0.768_bp1.93.hdf5'`
<!-- # Citation
```
@inproceedings{Tsai:2020:ABV,
        author = {Tsai, Yu-Ju and Liu, Yu-Lun and Ouhyoung, Ming and Chuang, Yung-Yu},
        title = {Attention-based View Selection Networks for Light-field Disparity Estimation},
        booktitle = {Proceedings of the 34th Conference on Artificial Intelligence (AAAI)},
        year = {2020}
}
``` -->

Last modified data: 2023/05/28.

The code is modified and heavily borrowed from LFattNet: <https://github.com/LIAGM/LFattNet>, SubFocal: <https://github.com/chaowentao/SubFocal>

The code they provided is greatly appreciated.



