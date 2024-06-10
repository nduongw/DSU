#!/bin/bash
DATA=./DATA
DATASET=nico
D1=autumn
D2=dim
D3=grass
D4=outdoor
D5=rock
D6=water
SEED=42
method=conststyle

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer ConstStyleTrainer \
--uncertainty 0.5 \
--source-domains ${D6} ${D2} ${D3} ${D4} ${D5} \
--target-domains ${D1} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D1} \
--num_cluster 5 \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer ConstStyleTrainer \
--uncertainty 0.5 \
--source-domains ${D6} ${D1} ${D3} ${D4} ${D5} \
--target-domains ${D2} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D2} \
--num_cluster 5 \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer ConstStyleTrainer \
--uncertainty 0.5 \
--source-domains ${D6} ${D2} ${D1} ${D4} ${D5} \
--target-domains ${D3} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D3} \
--num_cluster 5 \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer ConstStyleTrainer \
--uncertainty 0.5 \
--source-domains ${D6} ${D2} ${D3} ${D1} ${D5} \
--target-domains ${D4} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D4} \
--num_cluster 5 \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer ConstStyleTrainer \
--uncertainty 0.5 \
--source-domains ${D6} ${D2} ${D3} ${D4} ${D1} \
--target-domains ${D5} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D5} \
--num_cluster 5 \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer ConstStyleTrainer \
--uncertainty 0.5 \
--source-domains ${D1} ${D2} ${D3} ${D4} ${D5} \
--target-domains ${D6} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D6} \
--num_cluster 5 \
--resume false)

echo "Running scripts in parallel"
wait # This will wait until both scripts finish
echo "Script done running"