# DiscoGAN in PyTorch

PyTorch implementation of [Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/abs/1703.05192).

<img src="./assets/model.png" width="80%">


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

To train a model:

    $ python main.py --dataset=edges2shoes --num_gpu=1
    $ python main.py --dataset=YOUR_DATASET_NAME --num_gpu=4

To test a model (use your `load_path`):

    $ python main.py --load_path=logs/edges2shoes_2017-03-18_02-39-31 --is_train=False


## Results

### 1. Toy dataset

Result of samples from 2-dimentional Gaussian mixture models. [IPython notebook](./notebooks/DiscoGAN.ipynb)

**# iteration: 0**:

<img src="./assets/toy_before.png" width="40%">

**# iteration: 10000**:

<img src="./assets/toy_after.png" width="40%">


### 2. Edges2shoes dataset

**# iteration: 9600**:

`x_A` -> `G_AB(x_A)`

<img src="assets/edges2shoes_valid_x_B.png" width="40%"> <img src="assets/edges2shoes_x_BA_9600.png" width="40%">

`x_B` -> `G_BA(x_B)`

<img src="assets/edges2shoes_valid_x_A.png" width="40%"> <img src="assets/edges2shoes_x_AB_9600.png" width="40%">


### 3. Edges2handbags dataset

**# iteration: 9500**:

`x_A` -> `G_AB(x_A)`

<img src="assets/edges2handbags_valid_x_B.png" width="40%"> <img src="assets/edges2handbags_x_BA_9500.png" width="40%">

`x_B` -> `G_BA(x_B)`

<img src="assets/edges2handbags_valid_x_A.png" width="40%"> <img src="assets/edges2handbags_x_AB_9500.png" width="40%">


### 4. Facades dataset

**# iteration: 19450**:

`x_A` -> `G_AB(x_A)`

<img src="assets/facades_valid_x_B.png" width="40%"> <img src="assets/facades_x_BA_19450.png" width="40%">

`x_B` -> `G_BA(x_B)`

<img src="assets/facades_valid_x_A.png" width="40%"> <img src="assets/facades_x_AB_19450.png" width="40%">


### 5. Cityscapes dataset

**# iteration: 8350**:

`x_A` -> `G_AB(x_A)`

<img src="assets/cityscapes_valid_x_B.png" width="40%"> <img src="assets/cityscapes_x_BA_8350.png" width="40%">

`x_B` -> `G_BA(x_B)`

<img src="assets/cityscapes_valid_x_A.png" width="40%"> <img src="assets/cityscapes_x_AB_8350.png" width="40%">


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io)
