# zero-shot
#python main.py --method=zero_shot --model=gpt3.5 --dataset=gsm8k
#python main.py --method=zero_shot --model=gpt3.5 --dataset=aqua

# zero cot
#python main.py --method=zero_shot_cot --model=gpt3.5 --dataset=gsm8k
#python main.py --method=zero_shot_cot --model=gpt3.5 --dataset=aqua

# few shot
#python main.py --method=few_shot5 --model=gpt3.5 --dataset=gsm8k
#python main.py --method=few_shot5 --model=gpt3.5 --dataset=aqua

# generating - demonstrtaion + instruction
python demo_ins.py --method=zero_shot --model=gpt3.5 --dataset=gsm8k
#python demo_ins.py --method=zero_shot --model=gpt3.5 --dataset=aqua


# auto cot
#python main.py --method=few_shot_autocot5 --model=gpt3.5 --dataset=gsm8k
#python main.py --method=few_shot_autocot5 --model=gpt3.5 --dataset=aqua
#
#
## retrieving-instruction
#python instruction.py --method=zero_shot --model=gpt3.5 --dataset=aqua
python instruction.py --method=zero_shot --model=gpt3.5 --dataset=gsm8k