#!/bin/bash

for i in {0..9}; do
	python reproducibility/convnets.py --seed $i
done



