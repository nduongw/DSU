#!/bin/bash
DATA=./DATA
DATASET=vlcs
D1=caltech
D2=labelme
D3=pascal
D4=sun
SEED=42
method=baseline

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D1} ${D2} \
--target-domains ${D3} ${D4} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/verify3/${D1}_${D2}-${D3}_${D4} \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D1} \
--target-domains ${D3} ${D4} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/verify3/${D1}-${D3}_${D4} \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D2} \
--target-domains ${D3} ${D4} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/verify3/${D2}-${D3}_${D4} \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D1} ${D3} \
--target-domains ${D2} ${D4} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/verify3/${D1}_${D3}-${D2}_${D4} \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D1} \
--target-domains ${D2} ${D4} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/verify3/${D1}-${D2}_${D4} \
--resume false)

(CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root ${DATA} \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D3} \
--target-domains ${D2} ${D4} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/verify3/${D3}-${D2}_${D4} \
--resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D2} ${D3} \
# --target-domains ${D1} ${D4} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D2}_${D3}-${D1}_${D4} \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D2} \
# --target-domains ${D1} ${D4} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D2}-${D1}_${D4} \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D3} \
# --target-domains ${D1} ${D4} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D3}-${D1}_${D4} \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D3} ${D4} \
# --target-domains ${D1} ${D2} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D3}_${D4}-${D1}_${D2} \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D3} \
# --target-domains ${D1} ${D2} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D3}-${D1}_${D2} \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D4} \
# --target-domains ${D1} ${D2} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D4}-${D1}_${D2} \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D2} ${D4} \
# --target-domains ${D1} ${D3} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D2}_${D4}-${D1}_${D3} \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D2} \
# --target-domains ${D1} ${D3} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D2}-${D1}_${D3} \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D4} \
# --target-domains ${D1} ${D3} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D4}-${D1}_${D3} \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D1} ${D4} \
# --target-domains ${D2} ${D3} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D1}_${D4}-${D2}_${D3} \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D1} \
# --target-domains ${D2} ${D3} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D1}-${D2}_${D3} \
# --resume false)

# (CUDA_VISIBLE_DEVICES=0 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D4} \
# --target-domains ${D2} ${D3} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify3/${D4}-${D2}_${D3} \
# --resume false)

echo "Running scripts in parallel"
wait # This will wait until both scripts finish
echo "Script done running"
