#!/bin/bash
DATA=./DATA
DATASET=terrainc
D1='38'
D2='46'
D3='100'
D4='43'
SEED=42
method=baseline

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D2} ${D3} ${D4} \
--target-domains ${D1} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D1} \
--resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D1} ${D3} ${D4} \
# --target-domains ${D2} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/${D2} \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D2} ${D1} ${D4} \
# --target-domains ${D3} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/${D3} \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D2} ${D3} ${D1} \
# --target-domains ${D4} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/${D4} \
# --resume false)

echo "Running scripts in parallel"
wait # This will wait until both scripts finish
echo "Script done running"