#!/bin/bash
DATA=./DATA
DATASET=vlcs
D1=sun
D2=labelme
D3=pascal
D4=caltech
SEED=40
method=conststyle

# (CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D2} ${D3} ${D4} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify2/${D2}_${D3}_${D4}-${D1} \
# --resume false)

(CUDA_VISIBLE_DEVICES=1 python tools/train.py \
--root ${DATA} \
--trainer ConstStyleTrainer \
--uncertainty 0.5 \
--source-domains ${D2} \
--target-domains ${D4} \
--seed ${SEED} \
--dataset-config-file configs/datasets/dg/${DATASET}_cs.yaml \
--config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/verify2/${D3}-${D1} \
--cluster ot \
--num_clusters 3 \
--update_interval 25 \
--prob 0.5 \
--alpha 0.5 \
--wandb 0 \
--conststyle_type ver5 \
--resume false)

# (CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D2} ${D3} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify2/${D2}_${D3}-${D1} \
# --resume false)

# (CUDA_VISIBLE_DEVICES=1 python tools/train.py \
# --root ${DATA} \
# --trainer Vanilla \
# --uncertainty 0.5 \
# --source-domains ${D3} \
# --target-domains ${D1} \
# --seed ${SEED} \
# --dataset-config-file configs/datasets/dg/${DATASET}.yaml \
# --config-file configs/trainers/dg/vanilla/${DATASET}.yaml \
# --output-dir output/dg/${DATASET}/${method}/verify2/${D2}_${D3}-${D1} \
# --resume false)

echo "Running scripts in parallel"
wait # This will wait until both scripts finish
echo "Script done running"
