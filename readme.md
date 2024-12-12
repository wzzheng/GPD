# GPD-1: Generative Pre-training for Driving
### [Paper](https://arxiv.org/pdf/2412.08643)  | [Project Page](https://wzzheng.net/GPD)  | [Code](https://github.com/wzzheng/GPD) 

Check out our [Large Driving Model](https://github.com/wzzheng/LDM/) Series! 

> GPD-1: Generative Pre-training for Driving

> [Zixun Xie](https://github.com/rainyNighti)\*, [Sicheng Zuo](https://github.com/zuosc19)\*, [Wenzhao Zheng](https://wzzheng.net/)\* $\dagger$, [Yunpeng Zhang](https://scholar.google.com/citations?user=UgadGL8AAAAJ&hl=zh-CN&oi=ao), [Dalong Du](TODO), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/), [Shanghang Zhang](https://scholar.google.com/citations?user=voqw10cAAAAJ&hl=en)$\ddagger$

\* Equal contribution $\dagger$ Project leader $\ddagger$ Corresponding author

GPD-1 proposes a unified approach that seamlessly accomplishes multiple aspects of scene evolution, including scene simulation, traffic simulation, closed-loop simulation, map prediction, and motion planning, all without additional fine-tuning.

![teaser](./assets/images/demo.png)

## News
- **[2024/12/12]** Code released.
- **[2024/12/12]** Paper released on [arXiv](https://arxiv.org/abs/2412.08643).

## Demo

The pre-trained GPD-1 can accomplish various tasks without finetuning using different prompts.

### Scene Simulation

![demo](./assets/gifs/SceneSimulation.gif)

### Traffic Simulation

![demo](./assets/gifs/TrafficSimulation.gif)

### Closed-Loop Simulation

![demo](./assets/gifs/CLS.gif)

### Map Prediction

![demo](./assets/gifs/MapPredition.gif)

### Motion Planning

![demo](./assets/gifs/MotionPlanning.gif)

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
[PlanTF](https://github.com/jchengai/planTF)
[Sledge](https://github.com/autonomousvision/sledge)
[Navsim](https://github.com/autonomousvision/navsim)

## Citation

If you find this project helpful, please consider citing the following paper:
```
  @article{gpd-1,
    title={GPD-1: Generative Pre-training for Driving},
    author={Xie, Zixun and Zuo, Sicheng and Zheng, Wenzhao and Zhang, Yunpeng and Du, Dalong and Zhou, Jie and Lu, Jiwen and Zhang, Shanghang},
    journal={arXiv preprint arXiv:2412.08643},
    year={2024}
}
```
