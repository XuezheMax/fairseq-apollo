# Mega: Moving Average Equipped Gated Attention

## Official Repository
The [official repository](https://github.com/facebookresearch/mega) of [Mega](https://arxiv.org/abs/2209.10655) is hosted in the [GitHub of Meta AI Research](https://github.com/facebookresearch/)


## [Models Checkpoints](https://drive.google.com/drive/folders/1er9ZyeXHmvHPk_lvLExlx25yuHt_ok3l?usp=sharing)
Task | Description | # params | Download
---|---|---|---
`LRA` | Mega on LRA tasks | -- | [mega.lra.zip](https://drive.google.com/file/d/16waj3AslaTHuCxokXJFFuRwygi8P9Wd4/view?usp=sharing)
`WMT'14 (En-De)` | Mega-base on WMT'14 En-De | 67M | [meta.wmt14ende.base.zip]()
`WMT'14 (De-En)` | Mega-base on WMT'14 De-En | 67M | [meta.wmt14deen.base.zip]()
`SC-Raw` | Mega-base on raw Speech Commands | 300k | [meta.sc.base.zip](https://drive.google.com/file/d/1NANfdH_iBnliPfAwLlrc-B3_sJe_bM2V/view?usp=sharing)
`SC-Raw` | Mega-big on raw Speech Commands | 476k | [meta.sc.big.zip](https://drive.google.com/file/d/1NANfdH_iBnliPfAwLlrc-B3_sJe_bM2V/view?usp=sharing)
`WikiText-103` | Language modeling on WikiText-103 | 252M |[meta.wiki103.zip]()
`Enwiki8` | Language modeling on Enwiki8 | 39M | [meta.enwiki8.zip]()


## Experiments

- [Long Range Arena](README.lra.md)
- [Machine Translation](README.mt.md)
- [Speech Classification](README.sc.md)
- [Language Modeling](README.lm.md)
- [ImageNet](https://github.com/xuezhemax/mega-image)


## Citation

```bibtex
@article{ma2022mega,
  title={Mega: Moving Average Equipped Gated Attention},
  author={Ma, Xuezhe and Zhou, Chunting and Kong, Xiang and He, Junxian and Gui, Liangke and Neubig, Graham and May, Jonathan and Zettlemoyer, Luke},
  journal={arXiv preprint arxiv.2209.10655},
  year={2022}
}
```