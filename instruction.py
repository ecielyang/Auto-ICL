import argparse
import logging
import torch
import random
import time
import os
from utils import *


def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')

    fix_seed(args.random_seed)

    print("OPENAI_API_KEY:")
    print(os.getenv("OPENAI_API_KEY"))

    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder(args)
    answer_extractor = Decoder(args)

    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()

    if args.method == "few_shot":
        demo = create_demo_text2(args, cot_flag=False)
        demo_prompt = create_demo_text_prompt(args, cot_flag=False)
    elif args.method == "few_shot_cot":
        demo = create_demo_text(args, cot_flag=True)
        demo_prompt = create_demo_text_prompt(args, cot_flag=False)
    elif args.method == "few_shot_double":
        demo = create_demo_text(args, cot_flag=False) + create_demo_text2(args, cot_flag=False)
        demo_prompt = create_demo_text_prompt(args, cot_flag=False)
    elif args.method == "few_shot_double2":
        demo = create_demo_text2(args, cot_flag=False) + create_demo_text(args, cot_flag=False)
        demo_prompt = create_demo_text_prompt(args, cot_flag=False)
    else:
        pass



    total = 0
    correct_list = []
    logging.basicConfig(filename='final/'+args.model +'/log_'+args.method + '_onecot2/' + args.dataset + "_tem" + str(args.temperature) + "_p" + str(args.topp) + str(args.few_demo) + '.log', level=logging.INFO)
    for i, data in enumerate(dataloader):
        print('*************************')
        print("{}st data".format(i + 1))
        logging.info('*************************')
        logging.info("{}st data".format(i + 1))

        # Prepare question template ...
        original_prompt, y = data
        original_prompt = original_prompt[0]
        y = y[0].strip()

        plan = get_cot(args.dataset, args)

        new_prompt = "Chain of thought: " + plan + "\n" + "Question: " + original_prompt  + "\n" + "Solve the question step by step based on the chain of thought."
        new_prompt = "Q: " + new_prompt + "\nA: "

        if args.method == "zero_shot_cot":
            new_prompt =  new_prompt + args.cot_trigger

        print("New_prompt:")
        print(new_prompt)
        logging.info("New_prompt: " + "\n" + new_prompt)

        planned_answer = decoder.decode(args, new_prompt, 500, i, 2)
        rerun_time=0
        while planned_answer == "CANNOT_PROCESS":
            print(planned_answer + str(rerun_time))
            planned_answer = decoder.decode(args, new_prompt, 500, i, 2)
            rerun_time += 1
            if rerun_time == 3:
                break
        if rerun_time == 3:
            continue



        new_prompt2 = planned_answer + " " + args.direct_answer_trigger_for_zeroshot_cot
        max_length = args.max_length_direct
        print(new_prompt2)
        logging.info(new_prompt2)

        # output
        pred = answer_extractor.decode(args, original_prompt+new_prompt2, 32, i, 2)

        rerun_time=0
        while pred == "CANNOT_PROCESS":
            print(pred + str(rerun_time))
            pred = answer_extractor.decode(args, original_prompt+new_prompt2, 32, i, 2)
            rerun_time += 1
            if rerun_time == 3:
                break
        if rerun_time == 3:
            continue

        print("OUTPUT: ")
        print(pred)

        # Clensing of predicted answer ...
        pred = answer_cleansing(args, pred)

        # Choose the most frequent answer from the list ...
        print("pred : {}".format(pred))
        print("GT : " + y)
        print('*************************')
        logging.info("pred : {}".format(pred))
        logging.info("GT : " + y)
        logging.info('*************************')

        # Checking answer ...
        correct = (np.array([pred]) == np.array([y])).sum().item()
        correct_list.append(correct)
        total += 1  # np.array([y]).size(0)

        # Calculate accuracy ...
        accuracy = (sum(correct_list) * 1.0 / total) * 100
        print("accuracy : {}".format(accuracy))
        logging.info("accuracy : {}".format(accuracy))

        if (args.limit_dataset_size != 0) and ((i + 1) >= args.limit_dataset_size):
            break
            # raise ValueError("Stop !!")



def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument(
        "--api_log_file_name", type=str, default=None,
        help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]"
    )

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")

    parser.add_argument(
        "--dataset", type=str, default="aqua", help="dataset used for experiment"
    )

    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1],
                        help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")

    parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")

    parser.add_argument(
        "--model", type=str, default="gpt3",
        help="model used for decoding. Note that 'gpt3' are the smallest models."
    )

    parser.add_argument(
        "--method", type=str, default="zero_shot_cot", help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1,
        help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=128,
        help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32,
        help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=200,
        help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=3, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="temperrature of GPT"
    )
    parser.add_argument(
        "--topp", type=float, default=1, help="top p of GPT"
    )
    parser.add_argument(
        "--few_demo", type=bool, default=False, help="Whether usefew shot to rewrite prompt"
    )

    args = parser.parse_args()

    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (in MM/DD/YYYY) is"
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
    elif args.dataset == "TOMI1" or args.dataset == "TOMI2":
        args.dataset_path = "http://maartensap.com/neuralToM/ToMi-finalNeuralTOM.csv"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")

    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger

    args.direct_answer_trigger_for_fewshot = "The answer is"

    if args.cot_trigger_no == 1:
        args.cot_trigger = "Let's think step by step."
    elif args.cot_trigger_no == 2:
        args.cot_trigger = "We should think about this step by step."
    elif args.cot_trigger_no == 3:
        args.cot_trigger = "First,"
    elif args.cot_trigger_no == 4:
        args.cot_trigger = "Before we dive into the answer,"
    elif args.cot_trigger_no == 5:
        args.cot_trigger = "Proof followed by the answer."
    elif args.cot_trigger_no == 6:
        args.cot_trigger = "Let's think step by step in a realistic way."
    elif args.cot_trigger_no == 7:
        args.cot_trigger = "Let's think step by step using common sense and knowledge."
    elif args.cot_trigger_no == 8:
        args.cot_trigger = "Let's think like a detective step by step."
    elif args.cot_trigger_no == 9:
        args.cot_trigger = "Let's think about this logically."
    elif args.cot_trigger_no == 10:
        args.cot_trigger = "Let's think step by step. First,"
    elif args.cot_trigger_no == 11:
        args.cot_trigger = "Let's think"
    elif args.cot_trigger_no == 12:
        args.cot_trigger = "Let's solve this problem by splitting it into steps."
    elif args.cot_trigger_no == 13:
        args.cot_trigger = "The answer is after the proof."
    elif args.cot_trigger_no == 14:
        args.cot_trigger = "Let's be realistic and think step by step."
    else:
        raise ValueError("cot_trigger_no is not properly defined ...")

    return args


if __name__ == "__main__":
    main()


