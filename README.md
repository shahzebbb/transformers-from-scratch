# transformers-from-scratch

In this project, I code the transformer architecture from scratch using basic PyTorch.

I received significant help from Andrej Kaparthy's 'Let's Build GPT' repository which can be found here: https://github.com/karpathy/ng-video-lecture

The purpose of this project is to demonstrate that I understand the architecture of a transformer well enough to implement it myself.

## Files in this repo

The following files are contained in this repo:

* **data**: This folder contains any data used during training
    * **shakespeare.txt**: Contains the shakespeare text used to train the GPT-like model
* **models**: This folder contains various transformer models
    * **gpt.py**: Contains the code to define a GPT-like model
*  **transformer**: This folder contains the code for the various building blocks of a transformer
    * **utils.py**: Contains the code for basic building blocks of attention
    *  **encoder.py**: Contains code to define an EncoderBlock which can be used to build an encoder-only transformer like Bert
    *  **decoder.py**: Contains the code to define a DecoderBlock which can be used to build a decoder-only transformer like GPT
    *  **cross_attention_decoder.py**: Contains code to define a CrossAttentionDecoderBlock which is used to build the cross-attention mechanism in the original Transformer paper.
* **GPT-Shakespeare-Training.ipnyb**: A notebook which is designed to train a GPT-like model on shakespeare text

## Things to do:

* Update MultiHeadAttention in utils.py so that it can directly work with multiply heads without needing the Head Class
* Add an encoder-only model (Bert?) in models/ and train it on sentiment data
* Add a Seq2Seq model in models/ and train it on translation tasks
