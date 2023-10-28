# Auto-ICL: In-Context Learning without Human Supervision

The official implementation of `Auto-ICL: In-Context Learning without Human Supervision` .

## Installation
Make sure you have Python>=3.8 installed on your machine.
```
pip install torch==1.8.2+cu111 torchtext==0.9.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install -r requirements.txt
```

## Set OpenAI API key
```
# https://beta.openai.com/account/api-keys
export OPENAI_API_KEY=(YOUR OPENAI API KEY)
```


## Quick Start

### Auto-ICL (generating) (our proposal)
```
# demonstrtaion + instruction as context
python demo_ins.py --method=zero_shot --model=gpt3.5 --dataset=gsm8k
```

### Auto-ICL (retrieving) (our proposal)
```
# instruction as context
python instruction.py --method=zero_shot --model=gpt3.5 --dataset=gsm8k
```
### Zero-shot
```
python main.py --method=zero_shot --model=gpt3.5 --dataset=gsm8k
```
### Zero-CoT
```
python main.py --method=zero_shot_cot --model=gpt3.5 --dataset=gsm8k
```

### Auto-CoT
```
python main.py --method=few_shot_autocot5 --model=gpt3.5 --dataset=gsm8k
```

### Few-shot
```
python main.py --method=few_shot5 --model=gpt3.5 --dataset=gsm8k
```
### Instruction generation, demonstration generation are provided in the `juputer` directory
