# Symbol-LLM
This repo contains the official implementation of our paper:

**Symbol-LLM: Leverage Language Models for Symbolic System in Visual Human Activity Reasoning (NeurIPS 2023)**

Xiaoqian Wu, Yong-Lu Li*, Jianhua Sun, Cewu Lu*

[[project page]](https://mvig-rhos.com/symbol_llm)
[[paper]](https://openreview.net/pdf?id=RJq9bVEf6N)
[[arxiv]](https://arxiv.org/abs/2311.17365)


## Generate the Proposed Symbolic System
Given an activity, the proposed symbolic system prompts a LLM to generate broad-coverage symbols and rational rules.
It is implemented in [generate_rule.py](generate_rule.py).

## Visual Reasoning
With generated symbols and rules, we can use it to reason out activities in images.
We detail the experiments on HICO, with zero-shot CLIP as baseline.

First, download the [DATA](DATA/) folder from [this link](https://drive.google.com/drive/folders/1UtaSPzzt-hoTgrXnqOLMdRgDXBk_PJV9?usp=sharing), with generated rules and symbol predictions.
Then run [hico_clip+reason.ipynb](hico_clip+reason.ipynb) to get the result.

Alternatively, you can generate rules and predict symbols yourselves :)
- To generate rules, run [generate_rule.py](generate_rule.py). Note that the rules may differ because the evolution of GPT API and the sampling uncertainty. 
- To predict symbols, please refer to [hico_predict_symbols.py](hico_predict_symbols.py), where BLIP2 is used.

## Citation
If you find this work useful, please cite via:
```
@inproceedings{wu2023symbol,
  title={Symbol-LLM: Leverage Language Models for Symbolic System in Visual 
  Human Activity Reasoning},
  author={Wu, Xiaoqian and Li, Yong-Lu and Sun, Jianhua and Lu, Cewu},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```