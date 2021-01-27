<h1 align="center">Fairseq-Apollo</h1>

Fairseq with the Apollo optimizer. This folder is based on the [fairseq package](https://github.com/pytorch/fairseq). 

## Experimental Results on WMT-14 En-De

| Method     |  Test BLEU       |
| :--------  |  :-------------: |
| SGD        |  26.59 (0.07)    |
| Adam       |  27.84 (0.12)    |
| RAdam      |  28.15 (0.15)    |
| AdaBelief  |  28.14 (0.11)    |
| **Apollo** | **28.34 (0.10)** |

We use the Transformer-base models.
Some key hyper-parameters are listed in the following table.
We also provide a training [recipe](/recipe.md) for more details of the experiments.

**Transformer-base on WMT-14 En-De**

|  Method    |    lr      |  weight decay  |  decoupled weight decay |  eps  |    lr scheduler     |  warmup updates  |  init_lr  |  gradient clip  |
| :--------- | :--------: | :------------: | :---------------------: | :---: | :-----------------: | :--------------: | :-------: | :-------------: |
|  SGD       |   0.1      |      1e-6      |         False           |   NA  |      milestone      |       1000       |    1e-4   |      1.0        |
|  Adam      |   0.0005   |      1e-4      |         True            |  1e-8 |    inverse sqrt     |       4000       |    1e-7   |      1.0        |
|  RAdam     |   0.0005   |      1e-4      |         True            |  1e-8 |      milestone      |        0         |     NA    |      1.0        |
|  AdaBelief |   0.0005   |      1e-4      |         True            | 1e-16 |      milestone      |       1000       |    1e-7   |      1.0        |
|  Apollo    |   10.0     |      1e-8      |         False           |  1e-4 |      milestone      |       1000       |    0.01   |      1.0        |

