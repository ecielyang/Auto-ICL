import openai
import os

# Sentence Generator (Decoder) for GPT-3 ...
def decoder(args, input):
    # GPT-3 API allows each users execute the API within 60 times in a minute ...
    # time.sleep(1)

    # https://beta.openai.com/account/api-keys
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Specify engine ...
    if args.model == "gpt-3.5-turbo-0301":
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                messages=[{"role": "system",
                "content": "text completion"},{"role": "user", "content": input}],
                max_tokens=int(args.max_length),
                temperature=args.temperature, 
                top_p=args.temperature,
                stop=["Q: "]
            )
        return response["choices"][0]["message"]["content"]

    if args.model == "gpt4":
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system",
            "content": "text completion"},{"role": "user", "content": input}],
            max_tokens=int(args.max_length),
            temperature=args.temperature,  
            top_p=args.temperature,
            stop=["Q: "]
        )
        return response["choices"][0]["message"]["content"]

    elif args.model == "davinci002":
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=input,
            max_tokens=args.max_length,
            temperature=args.temperature,
            top_p=args.temperature,
            stop=None
        )
        return response["choices"][0]["text"]

    elif args.model == "davinci003":
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=input,
            max_tokens=args.max_length,
            temperature=args.temperature,
            top_p=args.temperature,
            stop=None
        )

        return response["choices"][0]["text"]

    elif args.model == "davinci001":
        response = openai.Completion.create(
            engine="text-davinci-001",
            prompt=input,
            max_tokens=args.max_length,
            temperature=args.temperature,
            top_p=args.temperature,
            stop=None
        )
        return response["choices"][0]["text"]

    elif args.model == "ada001":
        response = openai.Completion.create(
            engine="text-ada-001",
            prompt=input,
            max_tokens=args.max_length,
            temperature=args.temperature,
            top_p=args.temperature,
            stop=None
        )
        return response["choices"][0]["text"]

    elif args.model == "babbage001":
        response = openai.Completion.create(
            engine="text-babbage-001",
            prompt=input,
            max_tokens=args.max_length,
            temperature=args.temperature,
            top_p=args.temperature,
            stop=None
        )
        return response["choices"][0]["text"]

    elif args.model == "curie001":
        response = openai.Completion.create(
                engine="text-curie-001",
                prompt=input,
                max_tokens=args.max_length,
                temperature=args.temperature,
                top_p=args.temperature,
                stop=None
            )
        return response["choices"][0]["text"]
    else:
        raise ValueError("model is not properly defined ... you can include your model in decoder function")
