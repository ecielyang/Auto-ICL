from API import decoder
import random as random
import json
import pandas as pd


def generate_cot(args, original_prompt):
    propmpt_generator = "Q: " + original_prompt
    propmpt_generator = propmpt_generator + "\"\n" + "A: Let's think step by step."
    cot = decoder(args,  original_prompt)
    return cot


def combine_query_cot(question_ls, plan_ls):
    # sumamrize a general chain-of-thought by n examples
    input = "Q: "
    for i in range(len(question_ls)):
        input = input + "\n\n"
        input = input + "Question " + str(i + 1) + ": " + question_ls[i]
        input = input + "\n" + "Chain of thought " + str(i + 1) + ": " + plan_ls[i]

    # add trigger
    trigger = "Given the questions and chain of thoughts above, we wish to have a general chain of thought for all questions instead of one chain of thoughts for each question. Can you generate one?"
    input1 = input + "\n\n" + trigger

    input1 = input1 + "\n" + "A: "
    return input1


def Instruction_generate(args):
    # retireved_num: number of demonstrtaions to generate instruction, default=5

    # retirve queries from the same dataset
    if args.dataset in ["aqua", "gsm8k", "multialrith", "object_tracking", "cycle_letters", "TOMI1", "TOMI2", "socialIQA", "coin_flip", "last_letters"]:
        retrieved_queries, _ = data_reader(args)
    else:
        raise ValueError("Include your own dataset in `data_reader` function")

    # Generate COT for each query
    COTs = [generate_cot(args, i) for i in retrieved_queries]
    # Combine query and COTs
    instrutcion_template = combine_query_cot(retrieved_queries, COTs)
    instruction = decoder(args, instrutcion_template)
    return instruction


def Instruction_demo_generate(args, original_prompt):
    ### Get demo ###
    planning = "Q: Generate " + str(args.generated_num) + " questions with the same structure as the given question: "
    propmpt_generator = planning + original_prompt + "\n\n" + "A: "
    examples = decoder(args, original_prompt).strip().strip("\"")
    example_ls = examples.split("\n")

    demo_cot = ""
    demo_example = ""
    for exa in example_ls:
        prompt = "Q: " + exa + "\nAnswer: Let's think step by step."
        answer = decoder(args, prompt)
        demo_cot += "Question: " + exa + "\nChain of thought: Let's think step by step." + answer + "\n\n"
        demo_example += exa + "\nAnswer: Let's think step by step." + answer + "\n\n"

    ### Get instruction ###
    prompt_plan = demo_cot + "Given the questions and chain of thoughts above, generate a general chain of thought to help model to solve questions."
    cot = decoder(args, prompt_plan)
    new_prompt = "Q: Examples: " + demo_example + "Plan:" + cot + "\n\n" + "Question: " + original_prompt + "\n\n" + "Please solve the question step by step based on the provided examples and the plan."
    new_prompt = new_prompt + "\n\nA: "
    return new_prompt


def Auto_ICL(args, query):
    if args.method == "Auto-ICL-retrieving":
        insturction = Instruction_generate(args)
        new_prompt = "Chain of thought: " + insturction + "\n" + "Question: " + query + "\n" + "Solve the question step by step based on the chain of thought."
        new_prompt = "Q: " + new_prompt + "\nA: "

    elif args.method == "Auto-ICL-generating":
        new_prompt= Instruction_demo_generate(args, query)

    return new_prompt


def shuffleDict(d):
  keys = list(d.keys())
  random.shuffle(keys)
  [(key, d[key]) for key in keys]
  random.shuffle(keys)
  [(key, d[key]) for key in keys]
  random.shuffle(keys)
  keys = [(key, d[key]) for key in keys]
  #keys = d(keys)
  return dict(keys)

def data_reader(args):
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "aqua":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "(" + "(".join(json_res["options"])
                choice = choice.replace("(", " (").replace(")", ") ")
                choice = "Answer Choices:" + choice
                questions.append(json_res["question"].strip() + " " + choice)
                answers.append(json_res["correct"])

    elif args.dataset == "gsm8k":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(json_res["answer"].split("#### ")[-1])

    elif args.dataset == "commonsensqa":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                questions.append(json_res["question"]["stem"].strip() + " " + choice)
                answers.append(json_res["answerKey"])

    elif args.dataset in ("addsub", "multiarith", "singleeq"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["sQuestion"].strip()
                a = str(line["lSolutions"][0])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)

    elif args.dataset == "strategyqa":
        with open(args.dataset_path) as f:
            json_data = json.load(f)["examples"]
            for line in json_data:
                q = line["input"].strip()
                a = int(line["target_scores"]["Yes"])
                if a == 1:
                    a = "yes"
                else:
                    a = "no"
                questions.append(q)
                answers.append(a)

    elif args.dataset == "svamp":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)

    elif args.dataset == "object_tracking":
        # elif args.dataset == "object_tracking":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            if args.dataset == "bigbench_date":
                choice_index = ['A', 'B', 'C', 'D', 'E', 'F']
            elif args.dataset in ("object_tracking"):
                choice_index = ['A', 'B', 'C']
            else:
                raise ValueError("dataset is not properly defined ...")
            for line in json_data:
                q = line["input"].strip()
                if args.dataset == "bigbench_date":
                    choice = "Answer Choices:"
                    # Randomly shuffle the answer choice dictionary because the original answer is always A ...
                    choice_dic = shuffleDict(line["target_scores"])
                elif args.dataset == "object_tracking":
                    choice = "\nWhich choice is true ? Answer Choices:"
                    choice_dic = line["target_scores"]
                else:
                    raise ValueError("dataset is not properly defined ...")
                for i, key_value in enumerate(choice_dic.items()):
                    key, value = key_value
                    choice += " ("
                    choice += choice_index[i]
                    choice += ") "
                    choice += key
                    if value == 1:
                        a = choice_index[i]
                        # a = key
                q = q + " " + choice
                questions.append(q)
                answers.append(a)


    elif args.dataset in ("coin_flip", "last_letters"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            for line in json_data:
                q = line["question"]
                a = line["answer"]
                questions.append(q)
                answers.append(a)

    elif args.dataset == "cycle_letters":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                q = "Please unscramble the letters into a word, and write that word: " + json_res["context"]
                a = json_res["completion"]
                questions.append(q)
                answers.append(a)
    elif args.dataset == "socialIQA":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                # only include others
                if json_res["promptQuestionFocusChar"] == "o":
                    q = json_res['context'] + " " + json_res['question']
                    q += " Answer Choices: (A) " + json_res['answerA'] + " (B) " + json_res[
                        'answerB'] + " (C) " + \
                         json_res['answerC']
                    questions.append(q)
                    answers.append(json_res["label_letter"])
    elif args.dataset == "TOMI2":
        data = pd.read_csv(args.dataset_path)
        second_idx = [i for i in range(data["qOrder"].shape[0]) if data["qOrder"][i] == "second_order"]
        for i in second_idx:
            q = data["story"][i] + " " + data["question"][i]
            choice = []
            choice.append(data["answerMem"][i])
            choice.append(data["answerReal"][i])
            choice.append("Unknown")
            random.shuffle(choice)
            q = q + " Answer Choices: (A) " + choice[0] + " (B) " + choice[1] + " (C) " + choice[2]
            questions.append(q)

            if choice[0] == data["answer"][i]:
                answers.append("A")
            elif choice[1] == data["answer"][i]:
                answers.append("B")
            elif choice[2] == data["answer"][i]:
                answers.append("C")
    elif args.dataset == "TOMI1":
        data = pd.read_csv(args.dataset_path)
        second_idx = [i for i in range(data["qOrder"].shape[0]) if data["qOrder"][i] == "first_order"]
        for i in second_idx:
            q = data["story"][i] + " " + data["question"][i]
            choice = []
            choice.append(data["answerMem"][i])
            choice.append(data["answerReal"][i])
            choice.append("Unknown")
            random.shuffle(choice)
            q = q + " Answer Choices: (A) " + choice[0] + " (B) " + choice[1] + " (C) " + choice[2]
            questions.append(q)

            if choice[0] == data["answer"][i]:
                answers.append("A")
            elif choice[1] == data["answer"][i]:
                answers.append("B")
            elif choice[2] == data["answer"][i]:
                answers.append("C")
    else:
        raise ValueError("dataset is not properly defined ...")

    # q_len_list = []
    # for q in questions:
    #     q_len_list.append(len(q.split(" ")))
    # q_len_mean = mean(q_len_list)
    #
    # print("dataset : {}".format(args.dataset))
    # print("data size : {}".format(len(answers)))
    # print("average num of words for each sample : {}".format(q_len_mean))
    questions = random.sample(questions, args.retrieved_num)
    return questions, answers





