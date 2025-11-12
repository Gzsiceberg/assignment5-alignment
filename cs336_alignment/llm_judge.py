import openai
import dotenv
import os
import json
import typer
import asyncio
from tqdm.asyncio import tqdm as atqdm
from cs336_alignment.logger import setup_logging, print_and_log

POE_API_KEY = dotenv.get_key(".env", "POE_API_KEY")
client = openai.AsyncOpenAI(
    api_key=POE_API_KEY,  # Get this from poe.com/api_key
    base_url="https://api.poe.com/v1",
)

# Semaphore to control max concurrency
MAX_CONCURRENCY = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


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


async def evaluate_single_example(d0: dict, d1: dict) -> dict:
    """Evaluate a single example using the LLM judge with concurrency control."""
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

    async with semaphore:  # Control max concurrency
        response = await client.chat.completions.create(
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

    reply_content: str = response.choices[0].message.content  # type: ignore
    
    result = {
        "ranking": None,
        "error": None,
        "model_name_1": model_name_1,
        "model_name_2": model_name_2
    }
    
    try:
        ranking = eval(reply_content)
        result["ranking"] = ranking
        print_and_log(f"Ranking: {ranking}")
    except Exception as e:
        result["error"] = str(e)
        print_and_log(f"Error evaluating response: {e}")
        print_and_log(f"LLM Judge Response: {reply_content}")
    
    return result


async def alpaca_eval_async(data_path: str, limit: int = 0):
    """Async version of llm_judge that processes examples concurrently."""
    eval_model_outputs = json.load(open(data_path, "r"))
    reference_outputs = json.load(open("data/alpaca_reference.json", "r"))

    eval_model_outputs = eval_model_outputs[:limit] if limit > 0 else eval_model_outputs
    reference_outputs = reference_outputs[:limit] if limit > 0 else reference_outputs

    total_num = len(reference_outputs)
    print_and_log(f"Evaluating {total_num} examples with max concurrency {MAX_CONCURRENCY}...")
    
    # Create tasks for all examples
    tasks = [
        evaluate_single_example(d0, d1)
        for d0, d1 in zip(eval_model_outputs, reference_outputs)
    ]
    
    # Execute all tasks concurrently with progress bar
    results = []
    for coro in atqdm(asyncio.as_completed(tasks), total=total_num):
        result = await coro
        results.append(result)
    
    # Count wins
    model_wins_count = {}
    for result in results:
        if result["ranking"] is not None:
            for entry in result["ranking"]:
                model = entry["model"]
                rank = entry["rank"]
                if model not in model_wins_count:
                    model_wins_count[model] = 0
                if rank == 1:
                    model_wins_count[model] += 1

    print_and_log("Final model win counts:")
    for model, wins in model_wins_count.items():
        win_rate = wins / total_num
        print_and_log(f"{model}: {wins} wins ({win_rate*100:.2f}%)")
    
    with open("data/alpaca_eval_results.json", "w") as f:
        json.dump(results, f, indent=4)

app = typer.Typer()

@app.command()
def alpaca_eval(data_path: str = typer.Argument("data/alpaca_qwen2.5-math-1.5b.json", help="Path to JSON file with model outputs to evaluate"),
                limit: int = typer.Option(0, "-l", help="Limit the number of examples to evaluate")):
    """Wrapper function to run the async llm_judge."""
    asyncio.run(alpaca_eval_async(data_path, limit))

@app.command()
def safe_eval():
    print_and_log("This function is a placeholder for safe_eval functionality.")
    pass

if __name__ == "__main__":
    app()