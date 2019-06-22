#!/bin/bash


if [[ -z "${network}" ]]; then
    network='resnet18'
fi

if [[ -z "${vendor}" ]]; then
    vendor=amd
fi

REPORT=${network}_${vendor}

# fp32
# ----------------------------------------------------------------------------------------------------------------------

for i in {0..9}; do
	python reproducibility/convnets.py --init reproducibility/weights/cpu_0_${network}.init --seed $i --report  ${REPORT}_1.json
done

for i in {0..9}; do
	python reproducibility/convnets.py --seed $i --report ${REPORT}_2.json
done

for i in {0..9}; do
	python reproducibility/convnets.py --seed $i --report ${REPORT}_3.json --init reproducibility/weights/cpu_${i}_${network}.init
done


# fp16
# ----------------------------------------------------------------------------------------------------------------------

for i in {0..9}; do
	python reproducibility/convnets.py --opt-level O1 --init reproducibility/weights/cpu_0_${network}.init --seed $i --report ${REPORT}_fp16__1.json
done

for i in {0..9}; do
	python reproducibility/convnets.py --opt-level O1 --seed $i --report ${REPORT}_fp16_2.json
done

for i in {0..9}; do
	python reproducibility/convnets.py --seed $i --report ${REPORT}_fp16_3.json --init reproducibility/weights/cpu_${i}_${network}.init
done

# ----------------------------------------------------------------------------------------------------------------------
