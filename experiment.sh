models=("CNN" "GoogleNet" "AlexNet" "VGGNet")

for m in "${models[@]}"
do  
    echo "Traning $m"
    python3 train.py -m $m
done