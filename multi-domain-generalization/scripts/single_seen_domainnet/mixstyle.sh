#!/bin/bash
DATA=./DATA
DATASET=domain_net
D1=clipart
D2=painting
D3=quickdraw
D4=real
D5=sketch
SEED=42
method=mixstyle

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D1} \
--target-domains ${D2} ${D3} ${D4} ${D5} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_ms.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}${D2}_${D3}_${D4}_${D5} \
--reduce 1 \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D2} \
--target-domains ${D1} ${D3} ${D4} ${D5} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_ms.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D1}_${D3}_${D4}_${D5} \
--reduce 1 \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D3} \
--target-domains ${D1} ${D2} ${D4} ${D5} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_ms.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D1}_${D2}_${D4}_${D5} \
--reduce 1 \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D4} \
--target-domains ${D1} ${D2} ${D3} ${D5} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_ms.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D1}_${D2}_${D3}_${D5} \
--reduce 1 \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D5} \
--target-domains ${D1} ${D2} ${D3} ${D4} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_ms.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D1}_${D2}_${D3}_${D4} \
--reduce 1 \
--resume false)

echo "Running scripts in parallel"
wait # This will wait until both scripts finish
echo "Script done running"