import argparse
from template import Auto_ICL
from API import decoder
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="Auto-ICL")
    parser.add_argument(
        "--dataset", type=str, default=None, help="dataset used for experiment"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-3.5-turbo-0301", help="model used for decoding. Note that 'gpt3' are the smallest models."
    )
    parser.add_argument(
        "--method", type=str, default="Auto-ICL-generating", help="method", choices=["Auto-ICL-generating", "Auto-ICL-retrieving"]
    )
    parser.add_argument(
        "--max_length", type=int, default=1000,
        help="maximum length of output tokens by model"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="temperature of GPT"
    )
    parser.add_argument(
        "--topp", type=float, default=1, help="top p of GPT"
    )
    parser.add_argument(
        "--retrieved_num", type=float, default=5, help="Retrieved number of queries for insturction generation"
    )
    parser.add_argument(
        "--generated_num", type=float, default=5, help="Number of queries for insturction-demonstrtaion generation"
    )
    args = parser.parse_args()

    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the concatenated letters is"
    elif args.dataset == "cycle_letters":
        args.dataset_path = "./dataset/cycle_letters/cycle_letters_in_word.jsonl"
        args.direct_answer_trigger = "\nTherefore, the word is"
    elif args.dataset == "socialIQA":
        args.dataset_path = "./dataset/socialIQa/socialIWa_v1.4_dev_wDims.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "TOMI2" or args.dataset == "TOMI1":
        args.dataset_path = "http://maartensap.com/neuralToM/ToMi-finalNeuralTOM.csv"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")
    return args

def main():
    args = parse_arguments()

    print("OPENAI_API_KEY:")
    print(os.getenv("OPENAI_API_KEY"))

    original_query=input("Input your query: ")

    if args.method == "Auto-ICL-generating":
        print('************** Auto-ICL (generating) Prompt **************')
    elif args.method == "Auto-ICL-retrieving":
        print('************** Auto-ICL (generating) Prompt **************')

    new_qeury = Auto_ICL(args, original_query)
    print(new_qeury)

    print('********************** Model output **********************')
    answer = decoder(args, new_qeury)
    print(answer)


if __name__ == "__main__":
    main()