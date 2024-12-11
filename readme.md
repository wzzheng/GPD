# GPD: Generative Pre-training for Driving
### [Paper](TODO)  | [Project Page](https://wzzheng.net/GPD)  | [Code](https://github.com/wzzheng/GPD) 

> GPD: Generative Pre-training for Driving

> [Zixun Xie](https://github.com/rainyNighti)\*, [Sicheng Zuo](https://github.com/zuosc19)\*, [Wenzhao Zheng](https://wzzheng.net/)\*$\dagger$, [Yunpeng Zhang](https://scholar.google.com/citations?user=UgadGL8AAAAJ&hl=zh-CN&oi=ao), [Yunpeng Zhang](TODO), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/), [Shanghang Zhang](https://scholar.google.com/citations?user=voqw10cAAAAJ&hl=en)$\ddagger$
  <br>

\* Equal contribution $\dagger$ Project leader $\ddagger$ Corresponding author

GPD proposes a unified approach that seamlessly accomplishes multiple aspects of scene evolution, including scene generation, traffic simulation, closed-loop simulation, map prediction, and motion planning, all without the need for additional fine-tuning.

![teaser](./assets/images/demo.png)

## News
- **[2024/12/12]** Paper released on [arXiv](TODO).

## Demo

![demo](./assets/gifs/GPDdemo.gif)

## Overview
![overview](./assets/images/approach.png)
Our model adapts the GPT-like architecture for autonomous driving scenarios with two key innovations: 1) a 2D map scene tokenizer based on VQ-VAE that generates discrete, high-level representations of the 2D BEV map, and 2) a hierarchical quantization agent tokenizer to encode agent information. 
Using a scene-level mask, the autoregressive transformer predicts future scenes by conditioning on both ground-truth and previously predicted scene tokens during training and inference, respectively.


## Getting Started

<!-- Code is available at: [GaussianFormer](https://github.com/huang-yh/GaussianFormer) -->
Code will be available soon.

## Related Projects

Our work is inspired by these excellent open-sourced repos:
[planTF](https://github.com/jchengai/planTF)
[sledge](https://github.com/autonomousvision/sledge)

## Citation

If you find this project helpful, please consider citing the following paper:
```
TODO
```