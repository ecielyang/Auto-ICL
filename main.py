import argparse
import logging
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
    answer_extractor = Decoder_A(args)

    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()

    if args.method in ("few_shot5", "few_example_cot"):
        demo = demo_Q_A5(args.dataset)
    if args.method == "few_shot10":
        demo = demo_Q_A10(args.dataset)
    elif args.method == "few_shot_un10":
        demo = demo_Q_A_un10(args.dataset)
    elif args.method == "few_shot_un5":
        demo = demo_Q_A_un5(args.dataset)
    elif args.method == "few_shot_no10":
        demo = demo_Q_A_no10(args.dataset)
    elif args.method == "few_shot_Q10":
        demo = demo_Q10(args.dataset)
    elif args.method == "few_shot_Q5":
        demo = demo_Q5(args.dataset)
    elif args.method == "few_shot_cot":
        demo = create_demo_text(args.dataset)
    elif args.method == "few_shot_autocot5":
        demo = demo_autocot5(args.dataset, args)
    elif args.method == "few_shot_autocot3":
        demo = demo_autocot3(args.dataset)
    elif args.method == "few_shot_autocot10":
        demo = demo_autocot10(args.dataset)
    else:
        pass

    total = 0
    correct_list = []
    logging.basicConfig(filename="final/" + args.model +"/"+ args.method + '/' + args.dataset + '.log', level=logging.INFO)
    for i, data in enumerate(dataloader):
        print('*************************')
        print("{}st data".format(i + 1))
        logging.info('*************************')
        logging.info("{}st data".format(i + 1))

        # Prepare question template ...
        x, y = data
        y = y[0].strip()

        x = "Q: " + x[0].strip("=") + "\n" + "A: "

        if args.method == "zero_shot":
            x = x + " " + args.direct_answer_trigger_for_zeroshot
            x = x
        elif args.method == "zero_shot_cot":
            x = x + " " + args.cot_trigger
        elif args.method in ("few_example_cot", "few_shot_autocot5", "few_shot_autocot3", "few_shot_autocot10"):
            x = demo + "\n" + x + " " + args.cot_trigger
        elif args.method == "few_shot_no":
            print("x", x)
            x = demo + "\n" + x + " " + args.direct_answer_trigger_for_zeroshot
            print("x2", x)
        elif args.method in ["few_shot_un5", "few_shot_Q5", "few_shot5", "few_shot_un10", "few_shot_Q10", "few_shot10"]:
            x = demo + "\n" + x
        elif args.method == "few_shot_cot":
            x = demo + x
        else:
            raise ValueError("method is not properly defined ...")

        print("Answer prediction by generating text ...")

        logging.info("Answer prediction by generating text ...")
        max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
        z = decoder.decode(args, x, max_length, i, 1)
        rerun_time=0
        while z == "CANNOT_PROCESS":
            print(z + str(rerun_time))
            z = decoder.decode(args, x, max_length, i, 1)
            rerun_time += 1
            if rerun_time == 3:
                continue

        # Answer extraction for zero-shot-cot ...
        if args.method in ("zero_shot_cot", "few_example_cot", "few_shot_autocot5", "few_shot_autocot10", "few_shot_autocot3"):
            z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
            max_length = args.max_length_direct
            pred = decoder.decode(args, z2, max_length, i, 2)

            rerun_time = 0
            while pred == "CANNOT_PROCESS":
                print(pred + str(rerun_time))
                pred = answer_extractor.decode(args, z2, max_length, i, 2)
                rerun_time += 1
                if rerun_time == 3:
                    break

            print(z2)
            print(pred)
            logging.info(z2 + pred)
        else:
            pred = z
            print(x + pred)
            logging.info(x + pred)

        print("-------pred", pred)

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

        if (args.limit_dataset_size != 0) and ((i + 1) >= args.limit_dataset_size):
            break
            # raise ValueError("Stop !!")
        # Calculate accuracy ...
        accuracy = (sum(correct_list) * 1.0 / total) * 100
        print("accuracy : {}".format(accuracy))
        logging.info("accuracy : {}".format(accuracy))


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
        "--model", type=str, default="gpt3", help="model used for decoding. Note that 'gpt3' are the smallest models."
    )

    parser.add_argument(
        "--method", type=str, default="zero_shot_cot", help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1,
        help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=500,
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
        "--api_time_interval", type=float, default=2, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="temperature of GPT"
    )
    parser.add_argument(
        "--topp", type=float, default=1, help="top p of GPT"
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
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the concatenated letters is"
    elif args.dataset == "anli":
        args.dataset_path = "./dataset/ANLI/dev.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (True, False or Neither) is"
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

    # "Therefore, the answer ..." -> "The answer ..."
    if args.dataset == "cycle_letters" and "cot" not in args.method:
        trigger = ""
        args.direct_answer_trigger_for_zeroshot = ""
        args.direct_answer_trigger_for_zeroshot_cot = ""

        args.direct_answer_trigger_for_fewshot = ""
    else:
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