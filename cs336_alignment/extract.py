import regex as re
from tqdm import tqdm
import datasets
from typing import List, Any


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
STRIP_STRS = [
    ":",
    ".",
    "/",
    ",",
    "#",
    "?",
    "$",
    '"',
    "'",
    # "ки" is the delimeter for Math-Shepherd
    "к",
    "и",
    # LaTeX
    "\\(",
    "\\)",
    "\\[",
    "\\]",
]
NO_TRAILING_STRS = ["(", "[", "{", "\\"] + STRIP_STRS
NO_PRECEDING_PUNCS = ["!", ")", "]", "}", "\\\\"] + STRIP_STRS
PRM800K_ANS_PRRFIX = "# Answer"
GSM8K_ANS_PREFIX = "####"


def extract_gsm_answer(completion):
    """
    Extract the numerical answer after #### marker.
    Follows official code for normalization:
    https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    """
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


def clean_trailing(
    s: str,  # The input string.
) -> str:  # The cleaned string with trailing punctuation marks removed.
    """Removes trailing punctuation marks from a string."""
    s = str(s).strip()
    while s != "" and s[-1] in NO_TRAILING_STRS:
        s = s[:-1].strip()
    return s


def extract_boxed(resp: str) -> str:
    ans = resp.split("oxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for i_pre, c in enumerate(ans[1:]):
            if ans[i_pre] == "\\":
                a += c
                continue
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def norm_str2bool(s: str) -> bool | None:
    """Converts a string representation of a boolean value to its corresponding boolean value."""
    s = str(s).lower().strip().replace("noindent", "")
    if any(pos in s for pos in ["yes", "true"]):
        return True
    elif any(neg in s for neg in ["no", "false"]):
        return False
    else:
        return None


def extract_explicit_ans(resp_str: str) -> str | None:
    resp_str = clean_trailing(resp_str)
    # might be answer only
    if "herefore" in resp_str:
        resp_str = resp_str.split("herefore")[-1].strip()
    if GSM8K_ANS_PREFIX in resp_str:
        resp_str = resp_str.split(GSM8K_ANS_PREFIX)[-1].strip()
    if PRM800K_ANS_PRRFIX in resp_str:
        resp_str = resp_str.split(PRM800K_ANS_PRRFIX)[-1].strip()

    if "oxed{" in resp_str:
        resp = extract_boxed(resp_str)
    else:
        resp = resp_str

        # should be answer only
        if "is the ans" in resp:
            resp = re.split(r"(,|\.|\!\|?)", resp.split("is the ans")[-2].strip())[
                -1
            ].strip()
        elif "is our ans" in resp:
            resp = re.split(r"(,|\.|\!\|?)", resp.split("is our ans")[-2].strip())[
                -1
            ].strip()
        elif "answer is:" in resp:
            resp = resp.split("answer is:")[-1].strip()
        elif "answer is" in resp:
            resp = resp.split("answer is")[-1].strip()
        elif "answer:" in resp:
            resp = resp.split("answer:")[-1].strip()
        elif "answer :" in resp:
            resp = resp.split("answer :")[-1].strip()
        elif "statement" in resp:
            bool_resp = norm_str2bool(resp.split("is ")[-1].strip())
            if bool_resp is not None:
                return str(bool_resp)
        else:
            return None

        if resp.startswith("$") and resp.endswith("$"):
            resp = resp[1:-1]

    return resp


def extract_ans(resp_str: str, strict_extract: bool) -> str:
    """Extract answer segment from complete `resp`."""
    ans = extract_explicit_ans(resp_str)
    if ans is not None:
        return ans
    elif not strict_extract:
        # Speculate with the last latex formula
        matches = re.findall(
            r"(?:\$|\\\(|\\\[)([^\$]+)(?:\$|\\\(|\\\[)", resp_str, re.DOTALL
        )
        if len(matches) > 0:
            return matches[-1]
        # Speculate with the last number
        matches = re.findall(r"-?\d*\.?\d+", resp_str.replace(",", ""))
        if len(matches) > 0:
            return matches[-1]
    return ""  # Empty str if no answer is found


def preprocess_text(text: str) -> str:
    text = text.replace("\n", " ").strip()
    text = re.sub(r"([.\!\?,'/()])", r"\1", text)
    text = text.lower()
    return text


def extract_prompt_and_response(
    ds: datasets.Dataset, limit: int, offset: int
) -> tuple[List[str], List[str]]:
    from rich import print

    print(
        f"Generating prompts and ground truths with limit={limit} and offset={offset}"
    )
    prompt_templ = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {0}
Assistant: <think>"""
    prompts: List[str] = []
    responses: List[str] = []
    unique_queries: dict[str, int] = {}
    skip_queries: set[str] = set()
    query_to_index: dict[str, tuple[int, int]] = {}

    for data in tqdm(
        ds, desc="Generating prompts and ground truths", total=len(ds), leave=False
    ):
        question: str = data["query"]  # type: ignore
        q_key = preprocess_text(question)
        resp: str = data["response"]  # type: ignore
        if q_key in unique_queries:
            unique_queries[q_key] += 1
            if q_key in skip_queries:
                continue
            (index, resp_length) = query_to_index[q_key]
            if len(resp) < resp_length:
                # Update to a shorter response
                query_to_index[q_key] = (index, len(resp))
                responses[index] = resp
            continue
        else:
            unique_queries[q_key] = 1
            if len(skip_queries) < offset or len(prompts) >= limit:
                skip_queries.add(q_key)
            if q_key in skip_queries:
                continue
        full_prompt = prompt_templ.format(question)
        prompts.append(full_prompt)
        query_to_index[q_key] = (len(prompts) - 1, len(resp))
        responses.append(resp)
    min_response_length = min([len(resp) for resp in responses])
    max_response_length = max([len(resp) for resp in responses])
    print(f"Generated {len(prompts)} unique prompts.")
    print(f"Response length - min: {min_response_length}, max: {max_response_length}")

    import numpy as np

    list_occurrences = np.array(list(unique_queries.values()))
    mean_occurrence = list_occurrences.mean()
    max_occurrence = list_occurrences.max()
    min_occurrence = list_occurrences.min()
    std_occurrence = list_occurrences.std()
    print(f"offset queries: {len(skip_queries)}")
    print(
        f"Mean occurrence: {mean_occurrence:.2f}, Max occurrence: {max_occurrence}, Min occurrence: {min_occurrence}, Std occurrence: {std_occurrence:.2f}"
    )
    return prompts, responses


def extract_prompt_and_response_with_format(
    ds: datasets.Dataset, limit: int, offset: int
) -> tuple[list[str], list[str]]:
    prompts, responses = extract_prompt_and_response(ds, limit, offset)
    for t, resp in enumerate(responses):
        ans = extract_ans(resp, False)
        responses[t] = f"{resp} </think> <answer> {ans} </answer>"
    return prompts, responses
