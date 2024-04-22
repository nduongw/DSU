#!/bin/bash
DATA=./DATA
DATASET=digits_dg
D1=mnist
D2=mnist_m
D3=svhn
D4=syn
SEED=42
method=dsu

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D3} ${D4} \
--target-domains ${D1} ${D2} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_dsu.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D1}_${D2} \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D2} ${D4} \
--target-domains ${D1} ${D3} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_dsu.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D1}_${D3} \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D2} ${D3} \
--target-domains ${D1} ${D4} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_dsu.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D1}_${D4} \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D1} ${D4} \
--target-domains ${D2} ${D3} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_dsu.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D2}_${D3} \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D1} ${D3} \
--target-domains ${D2} ${D4} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_dsu.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D2}_${D4} \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D1} ${D2} \
--target-domains ${D3} ${D4} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_dsu.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D3}_${D4} \
--resume false)

echo "Running scripts in parallel"
wait # This will wait until both scripts finish
echo "Script done running"
