import openai
import dotenv
import os
import json
import typer
from tqdm import tqdm
from cs336_alignment.logger import setup_logging, print_and_log

POE_API_KEY = dotenv.get_key(".env", "POE_API_KEY")
client = openai.OpenAI(
    api_key=POE_API_KEY,  # Get this from poe.com/api_key
    base_url="https://api.poe.com/v1",
)


system_prompt = """You are a helpful assistant that ranks models by the quality of their answers."""

prompt_template = """I want you to create a leaderboard of different large language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be Python dictionaries.

Here is the prompt:
{instruction}

Here are the outputs of the models:
### model {model_name_1},
answer: {output_1}

### model {model_name_2},
answer: {output_2}

Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:
[
    {{"model": "<model-name>", "rank": <model-rank>}},
    {{"model": "<model-name>", "rank": <model-rank>}}
]

Your response must be a valid Python list and should contain nothing else because we will directly execute it in Python. Please provide the ranking that the majority of humans would give.
"""


def llm_judge(data_path: str = typer.Argument("data/alpaca_qwen2.5-math-1.5b.json", help="Path to JSON file with model outputs to evaluate")):
    eval_model_outputs = json.load(open(data_path, "r"))
    reference_outputs = json.load(open("data/alpaca_reference.json", "r"))

    total_num = len(reference_outputs)
    print_and_log(f"Evaluating {total_num} examples...")
    model_wins_count = {}
    for d0, d1 in tqdm(zip(eval_model_outputs, reference_outputs), total=total_num):
        instruction = d1["instruction"]
        output_1 = d1["output"]
        output_2 = d0["output"]
        model_name_1 = d1["generator"]
        model_name_2 = d0["generator"]

        prompt = prompt_template.format(
            instruction=instruction,
            output_1=output_1,
            output_2=output_2,
            model_name_1=model_name_1,
            model_name_2=model_name_2,
        )

        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.0,
            max_tokens=1024,
        )

        reply_content: str = response.choices[0].message.content # type: ignore
        try:
            ranking = eval(reply_content)
            print_and_log(f"Ranking: {ranking}")
            for entry in ranking:
                model = entry["model"]
                rank = entry["rank"]
                if model not in model_wins_count:
                    model_wins_count[model] = 0
                if rank == 1:
                    model_wins_count[model] += 1
        except Exception as e:
            print_and_log(f"Error evaluating response: {e}")
            print_and_log(f"LLM Judge Response: {reply_content}")
            continue

    print_and_log("Final model win counts:")
    for model, wins in model_wins_count.items():
        win_rate = wins / total_num
        print_and_log(f"{model}: {wins} wins ({win_rate*100:.2f}%)")

if __name__ == "__main__":
    typer.run(llm_judge)