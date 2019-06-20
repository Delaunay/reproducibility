
network='resnet18'
vendor=amd

for i in {0..9}; do
        python reproducibility/convnets.py --arch $network --init reproducibility/weights/cpu_0_$network.init --seed $i
done

mv report.json ${network}_${vendor}_1.json

for i in {0..9}; do
	python reproducibility/convnets.py --arch $network --seed $i
done

mv report.json ${network}_${vendor}_2.json

# ------
for i in {0..9}; do
	python reproducibility/convnets.py --arch $network --opt-level O1 --init reproducibility/weights/cpu_0_$network.init --seed $i
done

mv report.json ${network}_${vendor}_fp16_1.json

for i in {0..9}; do
	python reproducibility/convnets.py --arch $network --opt-level O1 --seed $i
done

mv report.json ${network}_${vendor}_fp16_2.json

