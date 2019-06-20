#!/bin/bash

for i in {0..9}; do
	python reproducibility/convnets.py --opt-level O1 --seed $i
done



