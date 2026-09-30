# Few-Shot Class-Incremental Learning for Encrypted Network Intrusion Detection

Official implementation of EP-Net (Extra-Prototypes aided Encrypted Network Intrusion Detection), a Few-Shot Class-Incremental Learning (FSCIL) method for encrypted network intrusion detection.

## Environment

- Python 3.11, CUDA 11.8
- Install dependencies:
```
pip install torch==2.6.0 torchvision==0.21.0 scikit-learn==1.6.1 numpy==2.2.3 pandas==2.2.3 matplotlib==3.10.1 seaborn==0.13.2 tqdm==4.67.1
```
- `scapy==2.6.1` is additionally required only for pre-processing raw pcap files.

## Datasets

1. Pre-process pcap traffic and extract the 6 metadata sequences (LEN, DIR, TWI, IAT, TTL, TFL) of each bi-flow, as introduced in the paper. Save the processed data in pkl files.

2. Add the processed pkl paths to the data dir in this project:
    ```python
    ./data/CICIDS2017/pktseq/process.py
    ...
    pkl_list = [
            ".../Monday-WorkingHours.pkl",
            ".../Tuesday-WorkingHours.pkl",
            ".../Wednesday-WorkingHours.pkl",
            ".../Thursday-WorkingHours.pkl",
            ".../Friday-WorkingHours.pkl",
        ]
    ...
    ```
    ```python
    ./data/CSECICIDS2018/pktseq/process.py
    ...
    pkl_list = [
            ".../Wednesday-14-02-2018.pkl",
            ".../Thursday-15-02-2018.pkl",
            ".../Friday-16-02-2018.pkl",
            ".../Tuesday-20-02-2018.pkl",
            ".../Wednesday-21-02-2018.pkl",
            ".../Thursday-22-02-2018.pkl",
            ".../Friday-23-02-2018.pkl",
            ".../Wednesday-28-02-2018.pkl",
            ".../Thursday-01-03-2018.pkl",
            ".../Friday-02-03-2018.pkl"
        ]
    ...
    ```

## Run

All hyper-parameters are fixed to the paper settings in `params.py` (including the dataset-specific total class number and lambda_cp). Simply run:

1. CIC-IDS2017
```
python main.py --dataset cicids2017
```
2. CSE-CIC-IDS2018
```
python main.py --dataset csecicids2018
```

Optional arguments: `--gpuseq` (GPU ids, default `0`), `--seed` (default `1`), `--repeat_n` (number of repeated experiments, default `10`).
