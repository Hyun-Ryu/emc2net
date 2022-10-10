# EMC²-Net

Official PyTorch implementation of the submitted paper "EMC²-Net: Joint Equalization and Modulation Classification based on Constellation Network".

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

## Train
### Phase 1: *Classifier pretraining*
```bash
python train_phase1_noise_cirriculum.py --root "YOUR OWN ROOT DIRECTORY"
```

### Phase 2: *Equalizer training*
```bash
python train_phase2.py --root "YOUR OWN ROOT DIRECTORY" --data_name "NAME OF DATASET" --exp_name "NAME OF EXPERIMENT"
```

### Phase 3: *Equalizer-classifier fine-tuning*
```bash
python train_phase3.py --root "YOUR OWN ROOT DIRECTORY" --data_name "NAME OF DATASET" --exp_name "NAME OF EXPERIMENT"
```

## Test
### Fading datasets: Rician, Rayleigh
```bash
python test_fading.py --root "YOUR OWN ROOT DIRECTORY" --data_name "NAME OF DATASET" --exp_name "NAME OF EXPERIMENT"
```

### Pretraining dataset: AWGN+PO
```bash
python test_awgnpo.py --root "YOUR OWN ROOT DIRECTORY"
```