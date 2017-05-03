mkdir ./checkpoint/sunny-rainy-beach-lr00002-run1
mkdir ./checkpoint/sunny-rainy-beach-lr0002-run1
mkdir ./checkpoint/sunny-rainy-beach-lr002-run1
python cyclegan.py -t 180 -c ./checkpoint/sunny-rainy-beach-lr00002-run2/model.ckpt -l 0.00002 -sl 1
python cyclegan.py -t 180 -c ./checkpoint/sunny-rainy-beach-lr0002-run2/model.ckpt -l 0.0002 -sl 1
python cyclegan.py -t 180 -c ./checkpoint/sunny-rainy-beach-lr002-run2/model.ckpt -l 0.002 -sl 1

