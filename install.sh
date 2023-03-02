python -m pip install paddlepaddle-gpu

cd ./dataset || exit
mkdir test
mkdir train
mkdir model
touch test.json
touch train.json
cd ..

git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd ./PaddleDetection || exit
pip install -r requirements.txt
python setup.py install
cd ..
