## Counterfactual Knowledge Distillation for Popularity Debiasing in LLM-based Recommendation

## Abstract
Generative large language model (LLM)-based recommender systems (LRS) can directly generate suitable items from user histories and have emerged as an important new paradigm in recommendation. However, LRS often exhibit severe popularity bias, resulting in excessive exposure of popular items. Existing methods typically perform debiasing according to the global popularity of items or tokens, which may overlook personalized user preferences. To address this issue, we propose Counterfactual Knowledge Distillation for Popularity Debiasing in LLM-based Recommendation (CKDLRec), which aims to mitigate popularity bias while preserving personalized recommendation capability.
CKDLRec consists of two main stages. In the first stage, counterfactual data are constructed by jointly considering item popularity and users' historical preferences, and a teacher model with reduced popularity influence is trained on the resulting data. In the second stage, the student model is trained on the original interaction data, and debiased knowledge from the teacher is transferred through knowledge distillation. Meanwhile, representation-level adversarial learning is incorporated to further reinforce debiased knowledge transfer and reduce the interference of popularity bias. Extensive experiments on three real-world datasets demonstrate that CKDLRec achieves favorable performance in recommendation accuracy, fairness, and diversity.

## Prepare the pretrained Hugging Face model Qwen2.5-1.5B-Instruct
Qwen2.5-1.5B-Instruct  
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

## Download the datasets
CDs, Movies, and Toys  
https://amazon-reviews-2023.github.io/

## How to Train with the CKDLRec Framework
Preprocess the data
```bash
python preprocess.py --dataset cds --ts_start 2017-09 --ts_end 2020-09 
python preprocess.py --dataset movies --ts_start 2018-09 --ts_end 2020-09
python preprocess.py --dataset toys --ts_start 2019-09 --ts_end 2020-09
```

Build item embeddings
```bash
python build_item_emb.py --dataset cds
python build_item_emb.py --dataset movies
python build_item_emb.py --dataset toys
```

Build counterfactual data
```bash
python build_cf_data.py --dataset cds
python build_cf_data.py --dataset toys
python build_cf_data.py --dataset movies
```

Train and evaluate
```bash
bash train_cf_sft.sh
bash train_ckdlrec.sh
bash inference.sh
bash evaluate.sh
```

## Baselines

Train BIGRec
```bash
DATASET=cds GPU=0 bash baseline/sft/train.sh
DATASET=cds GPU=0 bash baseline/sft/inference.sh
DATASET=toys bash baseline/sft/evaluate.sh
```

Train IFairLRS
```bash
DATASET=cds GROUP=pop GPU=0 bash baseline/IFairLRS/train.sh
DATASET=cds GROUP=pop GPU=0 bash baseline/IFairLRS/inference.sh
DATASET=movies GROUP=pop bash baseline/IFairLRS/evaluate.sh
```

Train D2LR
```bash
DATASET=cds GPU=0 bash baseline/D2LR/train.sh
DATASET=cds GPU=0 bash baseline/D2LR/inference.sh
DATASET=toys bash baseline/D2LR/evaluate.sh
```

Train Flower
```bash
DATASET=cds GPU=0 bash baseline/Flower/train.sh
DATASET=toys GPU=0 bash baseline/Flower/inference.sh
DATASET=cds bash baseline/Flower/evaluate.sh
```

Train SPRec
```bash
DATASET=cds GPU=0 bash baseline/SPRec/train.sh
DATASET=cds GPU=0 bash baseline/SPRec/inference.sh
DATASET=cds bash baseline/SPRec/evaluate.sh
```

Train D3
```bash
DATASET=toys GPU=0 bash baseline/DecodingMatters/train.sh
DATASET=cds GPU=0 bash baseline/DecodingMatters/inference.sh
DATASET=movies bash baseline/DecodingMatters/evaluate.sh
```
