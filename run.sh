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
	python -m reproducibility.convnets "$@" --opt-level O0 --seed $i --report ${REPORT}_fp32_2.json
done

for i in {0..9}; do
	python -m reproducibility.convnets "$@" --opt-level O0 --init reproducibility/weights/cpu_0_${network}.init --seed $i --report ${REPORT}_fp32_1.json
done

for i in {0..9}; do
	python -m reproducibility.convnets "$@" --opt-level O0 --init reproducibility/weights/cpu_${i}_${network}.init --seed $i --report ${REPORT}_fp32_3.json
done


# fp16
# ----------------------------------------------------------------------------------------------------------------------

for i in {0..9}; do
	python -m reproducibility.convnets "$@" --opt-level O1 --seed $i --report ${REPORT}_fp16_2.json
done

for i in {0..9}; do
	python -m reproducibility.convnets "$@" --opt-level O1 --init reproducibility/weights/cpu_0_${network}.init --seed $i --report ${REPORT}_fp16_1.json
done

for i in {0..9}; do
	python -m reproducibility.convnets "$@" --opt-level O1 --init reproducibility/weights/cpu_${i}_${network}.init --seed $i --report ${REPORT}_fp16_3.json
done

# ----------------------------------------------------------------------------------------------------------------------
