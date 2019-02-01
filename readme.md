# High Resolution Face Swap

**A face swap implementation with much more higher resolution result (128x128)**, this is a promoted and optimized *swap face application* based on deepfake tech. our implementation did those changes based on original *deepfakes* implementation:

- *deepfakes* only support 64x64 input, we make it **deeper** and can output 128x128 size;
- we proposed a new network called *SwapNet, SwapNet128*;
- we changed the pre-proccess step with input data (such as warp face), make it more clear;
- we make the dataset loader more efficient, load pair face data directly from 2 dir;
- we proposed a new **face outline replace** tech to get a much more combination result with
  original image, their differences are like below image.


we will continuely update this repo, and make face swap much more intuitive and simple, anyone can build there own face changing model. Here are some result for 128x128 higher resolution face swap:

<p align="center">
<img src="https://s2.ax1x.com/2019/02/01/k3uReA.png">    
</p>



We have train on trump-cage and fanbingbing-galgadot convert model. The result is not fully trained yet, but it shows a promising result, the face in most situation can works perfect!

final result on face swap directly from original big image:

<p align="center">
    <img src="https://s2.ax1x.com/2019/02/01/k30qoV.png">
</p>

<p align="center">
    <img src="https://s2.ax1x.com/2019/02/01/k3BCe1.png">
</p>
<p align="center">
	<img src="https://s2.ax1x.com/2019/02/01/k3cQTs.png">
</p>


<p align="center>
	<img src="https://s2.ax1x.com/2019/02/01/k3cQTs.png">
</p>
<p align="center">
	<img src="https://s2.ax1x.com/2019/02/01/k3c3Yq.png">
</p>




As you can see above, we can achieve **high resolution** and seamlessly combination with face transformation. final result on face swap directly from video (to be added soon):



## Dependencies

our face swap implementation need *alfred-py* which can installed with:

```
sudo pip3 install alfred-py
```

## Pretrained Model

We only provided pretrained model for 128x128 model, and it was hosted by StrangeAI (http://codes.strangeai.pro). For train from scratch, you can download the trump cage dataset from: https://anonfile.com/p7w3m0d5be/face-swap.zip .
For those already StrangeAI VIP membership users, you can download the whole codes and models from http://strangeai.pro . 


## Train & Predict

the run, simply using:

```
python3 predict.py
# train fanbingbing-galgadot face swap
python3 train_trump_cage_64x64.py
python3 train_fbb_gal_128x128.py
```

this will predict on a trump face and convert it into cage face.


## More Info

if you wanna be invited to our computer vision discussion wechat group, you can add me via wechat or found us at: http://strangeai.pro which is **the biggest AI codes sharing platform in China**.



## Note About FaceSwap

We have did some failure attempt and experiments lots of combination to produce a good result, here are some notes you need to know to build a face swap tech:

- Size is everything: we have try maximum 256x256 as input size, but it fails to swap face style between 2 faces;
- Warp preprocess does not really matter, we have also trying to remove warp preprocess step and directly using target images for train, it can also success train a face swap model, but for dataset augumentation, better to warp it and make some random transform;
- loss is not really matter. Just kick of train, and train about 15000 epochs, and you can get good result;
- For data preparing, better extract faces first using dlib or [alfred](http://github.com/jinfagang/alfred)




## Faceswap Datasets

Actually, we gathered a lot of faces datasets. beside the default one, you may also access them via Baidu cloud disk.



## Copyright

*FaceSwap* is a project opensourced under MIT license, all right reserved by StrangeAI authors. website: http://strangeai.pro