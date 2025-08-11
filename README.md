# This is our implementation for the submitted paper.

## Enviroments
* NVIDIA A100, Driver == 535.129
* Cuda == 12.2
* Pytorch==2.2.2

## Dataset
Download: https://drive.google.com/file/d/11diAIO8g1KYJScD3sAaAVAg6U6WL2lY1/view?usp=sharing
Unzip the dataset.zip into ./data

## How to train the model from scratch
Run main.py
python main.py --device=$your_device_id(default=0) --period=$test_period(default=bull) --hidn-rnn=$hidn_size(default=64) --heads-att=$attention_haed(default=8) --ln-rnn=$num_of_rnn_layers(default=4)


## Reproduction of stock prediction experiment
Reproduction.ipynb.


