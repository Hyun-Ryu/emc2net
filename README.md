# EMC²-Net

Official source codes for "EMC²-Net: Joint Equalization and Modulation Classification based on Constellation Network", ICASSP 2023.

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Requirements
```bash
python==3.7.4
torch==1.10.2+cu113
numpy==1.17.2
scipy==1.3.1
```

### Getting started

- Clone this repo:
```bash
git clone https://github.com/Hyun-Ryu/emc2net emc2net
cd emc2net
```

## How to build synthetic dataset
- All of the data-generating codes are written in MATLAB, saved in `data_generation` folder.
- For AWGN+PO dataset, run `dataset_generation_AWGNPO.m`.
- For Rican or Rayleigh dataset, run `dataset_generation_fading.m`.

## Train
### Phase 1: *Classifier pretraining*
```bash
python train_phase1_noise_cirriculum.py \
    --root "YOUR OWN ROOT DIRECTORY" \
    --data_name "NAME OF DATASET" \
    --exp_name "NAME OF EXPERIMENT"
```

### Phase 2: *Equalizer training*
```bash
python train_phase2.py \
    --root "YOUR OWN ROOT DIRECTORY" \
    --data_name "NAME OF DATASET" \
    --exp_name "NAME OF EXPERIMENT" \
    --pretrain_exp_name "NAME OF PHASE 1 EXPERIMENT"
```

### Phase 3: *Equalizer-classifier fine-tuning*
```bash
python train_phase3.py \
    --root "YOUR OWN ROOT DIRECTORY" \
    --data_name "NAME OF DATASET" \
    --exp_name "NAME OF EXPERIMENT" \
    --pretrain_exp_name "NAME OF PHASE 1 EXPERIMENT" \
    --phase2_exp_name "NAME OF PHASE 2 EXPERIMENT"
```

## Test
### Fading datasets: Rician, Rayleigh
```bash
python test_fading.py \
    --root "YOUR OWN ROOT DIRECTORY" \
    --data_name "NAME OF DATASET" \
    --exp_name "NAME OF EXPERIMENT"
```

### Pretraining dataset: AWGN+PO
```bash
python test_awgnpo.py \
    --root "YOUR OWN ROOT DIRECTORY" \
    --data_name "NAME OF DATASET" \
    --exp_name "NAME OF EXPERIMENT"
```
