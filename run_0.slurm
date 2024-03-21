#!/bin/sh
#SBATCH --job-name="twi_infer_t"
#SBATCH --output="twi_infer_0.%j.%N.out"
#SBATCH --account=bckz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH --time=16:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1

module load anaconda3_gpu
module load cuda
git clone https://github.com/ztxz16/fastllm
cd fastllm
mkdir build
cd build
cmake .. -DUSE_CUDA=ON
make -j
cd tools && python setup.py install
cd ..

git clone https://github.com/XiaoyuLiu198/spatial_personality.git
cd spatial_personality
mkdir results
mkdir twitter_prompts
mkdir ckpts
pip install -r requirements.txt

cd twitter_prompts
wget  --no-check-certificate 'https://docs.google.com/uc?export=download&id=1CUxYShXbIkC9qnxoT3lFFneDWZlCrqbn' -O post_sample_1199.csv
readlink -f post_sample_1199.csv

cd ..

cd ckpts
pip uninstall gdown
pip install gdown==4.6.0
gdown https://drive.google.com/uc?id=1JTkg-z211GusTeY-ZU-IIQcq7wy1ovDr
unzip conscientiousness_ckpts.zip

cd ..
python single.py --file /twitter_prompts/ --checkpoint /ckpts/content/drive/MyDrive/twitter_inference_data/ckpts/conscientiousness --destination /results/ --start 1199 --end 1200