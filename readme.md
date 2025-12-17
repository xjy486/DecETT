# DecETT: Accurate App Fingerprinting under Encrypted Tunnels via Dual Decouple-based Semantic Enhancement

🤗Welcome to the repository of DecETT, a novel App Fingerprinting (AF) model for encrypted tunnel traffic. 

❤️Since this paper has not yet been accepted, we provide an evaluation demo for interested reviewers to validate this work, including a trained model and part of our dataset.

## Requirements
📖Python 3.8.16

📖Pytorch 1.12.1+cu113

📖Numpy 1.24.2


## Data
💾Part of our dataset is released for evaluation. The data is stored at  `saved_data/v2ray_corr/data.pkl`.

## Usage

```bash run.sh```

🚇Before running run.sh, you should replace the `PHTHON_PATH` and `MAIN_FILE` as your own Python interpreter path and file path, respectively. 

## Results
📂The evaluation results can be found in `saved_results`. We provide an evaluation result in `saved_results/24-10-21_16-16-45_DRL_v2ray_corr_test` as an example.

## how to train
```
python main.py --mode train --dataset v2ray_corr --model DRL --epochs 100 --batch_size 64 --device cuda:0 --verbose --recon_loss --class_num 54 --label app_label --model_file test_model.pkl --max_packet_len 3000 --min_num_pkts 3 --max_num_pkts 200 --lr 0.001 --loss DRLLoss_GRL --valid_ratio 0.2 --test_ratio 0.2
```
