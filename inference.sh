rm -r output/
cd PaddleDetection || exit
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c ../configs/config.yml --infer_dir=../inputs \
-o weights=../model/CascadeRCNN.pdparams
mv output ../outputs
cd ..
