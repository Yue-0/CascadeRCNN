cd PaddleDetection || exit
python tools/export_model.py \
-c ../configs/config.yml \
--output_dir=../model/Cascade_RCNN \
-o weights=../model/CascadeRCNN.pdparams use_gpu=false
cd ..
