

def extract_answer(response: str) -> str | None:
    if "The correct answer is" not in response:
        return None
    groups = response.split("The correct answer is")
    if len(groups) < 2:
        return None
    answer_part = groups[1].strip()

    # Extract the first word or character after "The correct answer is"
    answer = answer_part.split()[0].strip().strip(".").strip(",")
    if answer in {"A", "B", "C", "D"}:
        return answer
    return None
