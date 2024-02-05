# LMX

This is the official code repo for the paper ["Language Model Crossover: Variation through Few-shot Prompting"](https://arxiv.org/abs/2302.12170) (https://arxiv.org/abs/2302.12170).

Language Model Crossover (LMX) is a simple yet effective method of generating solution variations using Large Language Models (LLMs).
Parent solutions (represented as text) are concatenated and fed into the LLM, which naturally produces variations, which are parsed as children.

![alt text](https://github.com/jal278/lmx/blob/main/images/updated_lmx_examples.png)

This repository contains notebooks and scripts for running LMX in the domains from the paper.
From these examples we hope you will see how easy it is to apply LMX to new domains :)

## Notebooks and Scripts
- **Binary Strings**: This [notebook](binary_strings.ipynb) applies LMX to the classic evolutionary algorithms domain of evolving strings of 1's and 0's.
- **QD Sentiment**: This [notebook](lmx_sentiment_demo.ipynb) uses LMX, coupled with MAP-Elites, to generate rephrasings that maintain the original meaning as well as possible, while covering the space of underlying sentiment.
- **Sodaracers**: This [script](sodaracers.py) uses LMX to generate a diverse array of locomoting virtual creatures represented as python code.
- **Image Generation**: This [script](stablediffusion.py) uses LMX to optimize prompts to image generation models to maxmimize desired image properties.
- **Symbolic Regression**: This [script](symbolic_regresion.py) uses LMX to discover compact mathematical expressions that model datasets.
- **EDA Comparison**: This [notebook](probability_experiment/plot_probability_diff.ipynb) compares the distribution of offspring produced by LMX to that of a classical Estimation of Distribution Algorithm (EDA).
- **The Effect of Parent Ordering**: These [scripts and notebook](order_experiment) analyze the impact of ordering parents in different ways when feeding them into LMX.

All of these notebooks and scripts use the huggingface transformers API, so the LLM can be easily swapped out for one that's better or more appropriate for your particular domain.

Please let us know if you have any questions. Enjoy!

_sample results of QD sentiment optimization_

![alt text](https://github.com/jal278/lmx/blob/main/images/sentiment.png)

_sample results of symbolic regression_

![alt text](https://github.com/jal278/lmx/blob/main/images/symbolic_regression.png)

_sample results of prompt optimization for image generation_

![alt text](https://github.com/jal278/lmx/blob/main/images/image_generation.png)
