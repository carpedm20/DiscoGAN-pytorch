# Code from https://github.com/phillipi/pix2pix/blob/master/datasets/download_dataset.sh
FILE=$1
URL=https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=./data/$FILE.tar.gz
TARGET_DIR=./data/$FILE/
wget -N $URL -O $TAR_FILE
mkdir $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE
