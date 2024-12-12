# GPD-1: Generative Pre-training for Driving
### [Paper](TODO)  | [Project Page](https://wzzheng.net/GPD)  | [Code](https://github.com/wzzheng/GPD) 

> GPD-1: Generative Pre-training for Driving

> [Zixun Xie](https://github.com/rainyNighti)\*, [Sicheng Zuo](https://github.com/zuosc19)\*, [Wenzhao Zheng](https://wzzheng.net/)\*$\dagger$, [Yunpeng Zhang](https://scholar.google.com/citations?user=UgadGL8AAAAJ&hl=zh-CN&oi=ao), [Dalong Du](TODO), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/), [Shanghang Zhang](https://scholar.google.com/citations?user=voqw10cAAAAJ&hl=en)$\ddagger$
  <br>

\* Equal contribution $\dagger$ Project leader $\ddagger$ Corresponding author

GPD-1 proposes a unified approach that seamlessly accomplishes multiple aspects of scene evolution, including scene generation, traffic simulation, closed-loop simulation, map prediction, and motion planning, all without the need for additional fine-tuning.

![teaser](./assets/images/demo.png)

## News
- **[2024/12/12]** Paper released on [arXiv](TODO).

## Demo

![demo](./assets/gifs/GPDdemo.gif)

## Overview
![overview](./assets/images/approach.png)
Our model adapts the GPT-like architecture for autonomous driving scenarios with two key innovations: 1) a 2D map scene tokenizer based on VQ-VAE that generates discrete, high-level representations of the 2D BEV map, and 2) a hierarchical quantization agent tokenizer to encode agent information. 
Using a scene-level mask, the autoregressive transformer predicts future scenes by conditioning on both ground-truth and previously predicted scene tokens during training and inference, respectively.


## Getting Started &#128640; <a name="gettingstarted"></a>

- [1. Installation and download](docs/1_installation_&_download.md)
- [2. Feature cache](docs/2_feature_cache.md) 
- [3. Train map tokenizer](docs/3_train_map_tokenizer.md)
- [4. Train GPD](docs/4_train_GPD.md)


## Related Projects

Our work is inspired by these excellent open-sourced repos:
[planTF](https://github.com/jchengai/planTF)
[sledge](https://github.com/autonomousvision/sledge)
[navsim](https://github.com/autonomousvision/navsim)

## Citation

If you find this project helpful, please consider citing the following paper:
```
TODO
```