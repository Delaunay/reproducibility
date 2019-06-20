#!/bin/bash

for i in {0..9}; do
	python reproducibility/convnets.py --opt-level O1 --init reproducibility/weights/cpu_0.init --seed $i
done



