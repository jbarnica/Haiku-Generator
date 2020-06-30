# LSTM Automatic Haiku Generation

This Haiku Generator uses an LSTM recurrent neural network to learn to produce 5-7-5 haikus based on human data. 

Completed in Winter 2019 as part of a final project for LING 442: Anatomy of Langauge Processing Systems at the University of Michigan.

## Data
This project trains on poems collected for two similar projects.

The poems in haiku.txt are from [Haikuzao](https://github.com/herval/creative_machines/tree/master/haikuzao/src/main/resources) <br>
The poems in haikus.csv are from [PoetRNN](https://github.com/sballas8/PoetRNN/blob/master/data/haikus.csv)

Word embeddings are GloVe pre-trained vectors, which can be downloaded [here](https://nlp.stanford.edu/projects/glove/). Alternative word embeddings could be used with minor modifications to the code.

## Examples
#### Some select examples from training

epoch 63, loss=37.522348308899204 <br>
5 | autumn fireplace <br>
7 | another violets surge <br>
5 | through a bird feeder <br>

epoch 65, loss=37.080916997470865 <br>
5 | value matter time <br>
7 | through rounded coffee blanket <br>
5 | a stray dog's taller <br>

epoch 89, loss=32.45372910828343 <br>
5 | overnight downpour <br>
7 | a raven's perspiration <br>
5 | on the scarecrow's eyes <br>

epoch 163, loss=23.889254108768764 <br>
5 | village perceptive <br>
7 | a seed seller catches teeth <br>
5 | from the cut birch tree <br>

epoch 203, loss=21.14558372439964 <br>
5 | freshly shaved water <br>
7 | a grey essence of cider <br>
5 | yet as earlier <br>
