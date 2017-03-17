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

### Toy dataset

Result of samples from 2-dimentional Gaussian mixture models.

**# iteration: 0**:

<img src="./assets/toy_before.png" width="40%">

**# iteration: 10000**:

<img src="./assets/toy_after.png" width="40%">


### Edges2shoes dataset

**# iteration: 1000**:

<img src="./assets/edges2shoes_1000.png"  width="40%">

**# iteration: 10000**:

(in progress)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io)
