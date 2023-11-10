# Auto-ICL: In-Context Learning without Human Supervision

The official implementation of `Auto-ICL: In-Context Learning without Human Supervision` .


## Set OpenAI API key
```
# https://beta.openai.com/account/api-keys
export OPENAI_API_KEY=(YOUR OPENAI API KEY)
```

## Quick Start

### Auto-ICL retriving:
```angular2html
python main.py --method=Auto-ICL-retrieving --dataset=gsm8k
```
Please specify `dataset` when using `Auto-ICL-retrieving` method. 
Import your own dataset for retrieving if your dataset are not included.

### Auto-ICL generating:
```angular2html
python main.py
```
