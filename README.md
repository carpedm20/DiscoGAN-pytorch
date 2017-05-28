# DiscoGAN in PyTorch

PyTorch implementation of [Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/abs/1703.05192).

<img src="./assets/model.png" width="80%">

**\* All samples in README.md are genearted by neural network except the first image for each row.**  
\* Network structure is slightly diffferent ([here](https://github.com/carpedm20/DiscoGAN-pytorch/blob/master/models.py#L13-L32)) from the author's [code](https://github.com/SKTBrain/DiscoGAN/blob/master/discogan/model.py#L69-L125).


## Requirements

- Python 2.7
- [Pillow](https://pillow.readthedocs.io/en/4.0.x/)
- [tqdm](https://github.com/tqdm/tqdm)
- [PyTorch](https://github.com/pytorch/pytorch)
- [torch-vision](https://github.com/pytorch/vision)


## Usage

First download datasets (from [pix2pix](https://github.com/phillipi/pix2pix)) with:

    $ bash ./data/download_dataset.sh dataset_name

- `facades`: 400 images from [CMP Facades dataset](http://cmp.felk.cvut.cz/~tylecr1/facade/).
- `cityscapes`: 2975 images from the [Cityscapes training set](https://www.cityscapes-dataset.com/).
- `maps`: 1096 training images scraped from Google Maps
- `edges2shoes`: 50k training images from [UT Zappos50K dataset](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/).
- `edges2handbags`: 137K Amazon Handbag images from [iGAN project](https://github.com/junyanz/iGAN).

or you can use your own dataset by placing images like:

    data
    ├── YOUR_DATASET_NAME
    │   ├── A
    │   |   ├── xxx.jpg (name doesn't matter)
    │   |   ├── yyy.jpg
    │   |   └── ...
    │   └── B
    │       ├── zzz.jpg
    │       ├── www.jpg
    │       └── ...
    └── download_dataset.sh

**All images in each dataset should have same size** like using [imagemagick](https://www.imagemagick.org/script/index.php):

    # for Ubuntu
    $ sudo apt-get install imagemagick
    $ mogrify -resize 256x256! -quality 100 -path YOUR_DATASET_NAME/A/*.jpg
    $ mogrify -resize 256x256! -quality 100 -path YOUR_DATASET_NAME/B/*.jpg

    # for Mac
    $ brew install imagemagick
    $ mogrify -resize 256x256! -quality 100 -path YOUR_DATASET_NAME/A/*.jpg
    $ mogrify -resize 256x256! -quality 100 -path YOUR_DATASET_NAME/B/*.jpg

    # for scale and center crop
    $ mogrify -resize 256x256^ -gravity center -crop 256x256+0+0 -quality 100 -path ../A/*.jpg

To train a model:

    $ python main.py --dataset=edges2shoes --num_gpu=1
    $ python main.py --dataset=YOUR_DATASET_NAME --num_gpu=4

To test a model (use your `load_path`):

    $ python main.py --dataset=edges2handbags --load_path=logs/edges2handbags_2017-03-18_10-55-37 --num_gpu=0 --is_train=False


## Results

### 1. Toy dataset

Result of samples from 2-dimensional Gaussian mixture models. [IPython notebook](./notebooks/DiscoGAN.ipynb)

**# iteration: 0**:

<img src="./assets/toy_before.png" width="30%">

**# iteration: 10000**:

<img src="./assets/toy_after.png" width="30%">


### 2. Shoes2handbags dataset

**# iteration: 11200**:

`x_A` -> `G_AB(x_A)` -> `G_BA(G_AB(x_A))` (shoe -> handbag -> shoe)

<img src="assets/shoes2handbags_valid_x_B.png" width="30%"> <img src="assets/shoes2handbags_x_BA_11200.png" width="30%"> <img src="assets/shoes2handbags_x_BAB_11200.png" width="30%">

`x_B` -> `G_BA(x_B)` -> `G_AB(G_BA(x_B))` (handbag -> shoe -> handbag)

<img src="assets/shoes2handbags_valid_x_A.png" width="30%"> <img src="assets/shoes2handbags_x_AB_11200.png" width="30%"> <img src="assets/shoes2handbags_x_ABA_11200.png" width="30%">

`x_A` -> `G_AB(x_A)` -> `G_BA(G_AB(x_A))` -> `G_AB(G_BA(G_AB(x_A)))` -> `G_BA(G_AB(G_BA(G_AB(x_A))))` -> ...

<img src="assets/shoes2handbags_repetitive_0_x_A_0.png" width="13%"> <img src="assets/shoes2handbags_repetitive_0_x_A_1.png" width="13%"> <img src="assets/shoes2handbags_repetitive_0_x_A_2.png" width="13%"> <img src="assets/shoes2handbags_repetitive_0_x_A_3.png" width="13%"> <img src="assets/shoes2handbags_repetitive_0_x_A_4.png" width="13%"> <img src="assets/shoes2handbags_repetitive_0_x_A_5.png" width="13%"> <img src="assets/shoes2handbags_repetitive_0_x_A_6.png" width="13%">


### 3. Edges2shoes dataset

**# iteration: 9600**:

`x_A` -> `G_AB(x_A)` -> `G_BA(G_AB(x_A))` (color -> sketch -> color)

<img src="assets/edges2shoes_valid_x_B.png" width="30%"> <img src="assets/edges2shoes_x_BA_9600.png" width="30%"> <img src="assets/edges2shoes_x_BAB_9600.png" width="30%">

`x_B` -> `G_BA(x_B)` -> `G_AB(G_BA(x_B))` (sketch -> color -> sketch)

<img src="assets/edges2shoes_valid_x_A.png" width="30%"> <img src="assets/edges2shoes_x_AB_9600.png" width="30%"> <img src="assets/edges2shoes_x_ABA_9600.png" width="30%">

`x_A` -> `G_AB(x_A)` -> `G_BA(G_AB(x_A))` -> `G_AB(G_BA(G_AB(x_A)))` -> `G_BA(G_AB(G_BA(G_AB(x_A))))` -> ...

<img src="assets/edges2shoes_repetitive_0_x_A_0.png" width="13%"> <img src="assets/edges2shoes_repetitive_0_x_A_1.png" width="13%"> <img src="assets/edges2shoes_repetitive_0_x_A_2.png" width="13%"> <img src="assets/edges2shoes_repetitive_0_x_A_3.png" width="13%"> <img src="assets/edges2shoes_repetitive_0_x_A_4.png" width="13%"> <img src="assets/edges2shoes_repetitive_0_x_A_5.png" width="13%"> <img src="assets/edges2shoes_repetitive_0_x_A_6.png" width="13%">


### 4. Edges2handbags dataset

**# iteration: 9500**:

`x_A` -> `G_AB(x_A)` -> `G_BA(G_AB(x_A))` (color -> sketch -> color)

<img src="assets/edges2handbags_valid_x_B.png" width="30%"> <img src="assets/edges2handbags_x_BA_9500.png" width="30%"> <img src="assets/edges2handbags_x_BAB_9500.png" width="30%">

`x_B` -> `G_BA(x_B)` -> `G_AB(G_BA(x_B))` (sketch -> color -> sketch)

<img src="assets/edges2handbags_valid_x_A.png" width="30%"> <img src="assets/edges2handbags_x_AB_9500.png" width="30%"> <img src="assets/edges2handbags_x_ABA_9500.png" width="30%">

`x_A` -> `G_AB(x_A)` -> `G_BA(G_AB(x_A))` -> `G_AB(G_BA(G_AB(x_A)))` -> `G_BA(G_AB(G_BA(G_AB(x_A))))` -> ...

<img src="assets/edges2handbags_repetitive_0_x_A_0.png" width="13%"> <img src="assets/edges2handbags_repetitive_0_x_A_1.png" width="13%"> <img src="assets/edges2handbags_repetitive_0_x_A_2.png" width="13%"> <img src="assets/edges2handbags_repetitive_0_x_A_3.png" width="13%"> <img src="assets/edges2handbags_repetitive_0_x_A_4.png" width="13%"> <img src="assets/edges2handbags_repetitive_0_x_A_5.png" width="13%"> <img src="assets/edges2handbags_repetitive_0_x_A_6.png" width="13%">


### 5. Cityscapes dataset

**# iteration: 8350**:

`x_B` -> `G_BA(x_B)` -> `G_AB(G_BA(x_B))` (image -> segmentation -> image)

<img src="assets/cityscapes_valid_x_A.png" width="30%"> <img src="assets/cityscapes_x_AB_8350.png" width="30%"> <img src="assets/cityscapes_x_ABA_8350.png" width="30%">

`x_A` -> `G_AB(x_A)` -> `G_BA(G_AB(x_A))` (segmentation -> image -> segmentation)

<img src="assets/cityscapes_valid_x_B.png" width="30%"> <img src="assets/cityscapes_x_BA_8350.png" width="30%"> <img src="assets/cityscapes_x_BAB_8350.png" width="30%">


### 6. Map dataset

**# iteration: 22200**:

`x_B` -> `G_BA(x_B)` -> `G_AB(G_BA(x_B))` (image -> segmentation -> image)

<img src="assets/maps_valid_x_A.png" width="30%"> <img src="assets/maps_x_AB_22200.png" width="30%"> <img src="assets/maps_x_ABA_22200.png" width="30%">

`x_A` -> `G_AB(x_A)` -> `G_BA(G_AB(x_A))` (segmentation -> image -> segmentation)

<img src="assets/maps_valid_x_B.png" width="30%"> <img src="assets/maps_x_BA_22200.png" width="30%"> <img src="assets/maps_x_BAB_22200.png" width="30%">


### 7. Facades dataset

Generation and reconstruction on dense segmentation dataset looks weird which are not included in the paper.  
I guess a naive choice of `mean square error` loss for reconstruction need some change on this dataset.

**# iteration: 19450**:

`x_B` -> `G_BA(x_B)` -> `G_AB(G_BA(x_B))` (image -> segmentation -> image)

<img src="assets/facades_valid_x_A.png" width="30%"> <img src="assets/facades_x_AB_19450.png" width="30%"> <img src="assets/facades_x_ABA_19450.png" width="30%">

`x_A` -> `G_AB(x_A)` -> `G_BA(G_AB(x_A))` (segmentation -> image -> segmentation)

<img src="assets/facades_valid_x_B.png" width="30%"> <img src="assets/facades_x_BA_19450.png" width="30%"> <img src="assets/facades_x_BAB_19450.png" width="30%">


## Related works

- [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)
- [BEGAN-tensorflow](https://github.com/carpedm20/BEGAN-tensorflow)
- [simulated-unsupervised-tensorflow](https://github.com/carpedm20/simulated-unsupervised-tensorflow)



## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io)
