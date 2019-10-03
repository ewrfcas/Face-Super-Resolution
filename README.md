# Face-Super-Resolution

Face super resolution based on ESRGAN (https://github.com/xinntao/BasicSR)

## Results

INPUT & AFTER-SR & GROUND TRUTH

![result](results/result.png)

## Usage

### Train

1. Run python gen_lr_imgs.py to get the face imgs with low resolution and pool qualities

2. Set the dir in train.py

> hr_path: The path list of imgs with high resolution.

> lr_path: The path of imgs with low resolution.

3. Run python train.py

### Pretrained models

1. Dlib alignment shape_predictor_68_face_landmarks.dat 

(https://pan.baidu.com/s/19Y-AYnXs6ubIh4vlkyvqbQ) 

(https://drive.google.com/open?id=1u3h3nX5f_w-HJV8Nd1zwqc3uTnVja5Ol)

2. Generator weights 

90000_G.pth 

(https://pan.baidu.com/s/14ITkNz_t0E7hRv0-tTAjhA) 

(https://drive.google.com/open?id=1CZkLZPtbJepgksCM93MvsY7NgqnEZSvk)

> 90000_G.pth (The last activation in G is linear, clearer)

200000_G.pth 

(https://pan.baidu.com/s/1Osge_4JjPyvG5Xfnbe9KVA) 

(https://drive.google.com/open?id=1B6BQu5Qk8eIu8MGTWJHnJxaxY1zCqQEt)

> 200000_G.pth (The last activation in G is tanh)

### Test

1. Download 'shape_predictor_68_face_landmarks.dat' and '90000_G.pth'

2. Set 'pretrain_model_G' in test.py

3. RUN python test.py

![conduct](results/AI日读.jpg)



