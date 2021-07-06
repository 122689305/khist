#!/usr/bin/env sh
data=$1
wget https://sdv-datasets.s3.amazonaws.com/$data.zip
unzip $data.zip
rm $data.zip
