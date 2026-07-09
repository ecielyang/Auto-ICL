# Auto-ICL: In-Context Learning without Human Supervision

Official implementation of the paper **"Auto-ICL: In-Context Learning without Human Supervision"**.

📄 [Paper (arXiv:2311.09263)](https://arxiv.org/abs/2311.09263)

## Overview

Standard in-context learning (ICL) relies on human-provided context, such as labeled examples and explicit instructions, which is labor-intensive and limits models to tasks that humans can readily annotate. **Auto-ICL** is a framework that lets a large language model autonomously generate its own examples and instructions for problem-solving, removing the need for human supervision.

The framework supports two modes:

- **Auto-ICL (retrieving)** — retrieves related queries from the dataset and applies a shared instruction across all queries in that dataset.
- **Auto-ICL (generating)** — generates examples and instructions tailored to each individual query, with no dataset retrieval.

Across a range of models and datasets, model-generated contexts outperform human-annotated contexts (including Few-Shot and Few-Shot-CoT) as well as prior self-generated approaches such as Zero-CoT and Auto-CoT.

## Setup

Set your OpenAI API key as an environment variable ([get a key here](https://beta.openai.com/account/api-keys)):

```bash
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
```

## Quick Start

### Auto-ICL (retrieving)

```bash
python main.py --method=Auto-ICL-retrieving --dataset=gsm8k
```

The `--dataset` argument is required for the retrieving method. To use a dataset that is not already included, import your own for retrieval.

### Auto-ICL (generating)

```bash
python main.py
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{yang2023auto,
  title={Auto-icl: In-context learning without human supervision},
  author={Yang, Jinghan and Ma, Shuming and Wei, Furu},
  journal={arXiv preprint arXiv:2311.09263},
  year={2023}
}
```
