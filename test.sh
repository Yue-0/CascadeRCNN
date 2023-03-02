cd PaddleDetection || exit
export CUDA_VISIBLE_DEVICES=0
python tools/eval.py -c ../configs/config.yml \
-o weights=../model/CascadeRCNN.pdparams
rm bbox.json
cd ..
