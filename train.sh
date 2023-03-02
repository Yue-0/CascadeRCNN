cd PaddleDetection || exit
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c ../configs/config.yml
mv output/config/model_final.pdparams ../model/CascadeRCNN.pdparams
rm -r output
cd ..
