# Luna: Linear Unified Nested Attention

https://arxiv.org/abs/2106.01540

## Intrudoction
Luna is an efficient attention mechanism, which uses two nested attention functions to approximate the regular softmax attention in Transformer.
See the associated paper for more details.

## Pre-trained models
Model | Description | # params | Download
---|---|---|---
`luna.base.tied` | Luna with sharing key and value parameters | 132M | [luna.base.tied.tar.gz]()
`luna.base.untied` | Luna without sharing key and value parameters | 147M | [luna.base.untied.tar.gz]()


## Experiments

- [Long Range Arena](README.lra.md)
- [WMT14 English-German](README.wmt.md)
- [Finetuning on GLUE](README.glue.md)
- [Finetuning on RACE](README.race.md)
- [Finetuning on Commonsense QA (CSQA)](README.csqa.md)


## Citation

```bibtex
@inproceedings{ma2021luna,
  title={Luna: Linear Unified Nested Attention},
  author={Ma, Xuezhe and Kong, Xiang and Wang, Sinong and Zhou, Chunting and May, Jonathan and Ma, Hao and Zettlemoyer, Luke},
  booktitle = {Advances in Neural Information Processing Systems},
  publisher = {Curran Associates, Inc.},
  year={2021}
}
```
