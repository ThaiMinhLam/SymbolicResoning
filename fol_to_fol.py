import json
import re


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def extract_predicates(sample: dict):
    premises_nl = sample.get("premises-NL", [])
    questions_nl = sample.get("question", [])
    LLM_fol = sample.get("LLM-FOL", [])
    questions_fol = sample.get("question-FOL", [])

    all_nl = premises_nl + questions_nl
    all_statements = LLM_fol + questions_fol

    predicates_per_statement = []

    for stmt in all_statements:
        temp = []
        pred_matches = re.findall(r'([a-zA-Z_]+)\(([^)]+)\)', stmt)
        for pred_name, args in pred_matches:
            temp.append(f"{pred_name}({args})")
        predicates_per_statement.append(temp)

    return all_nl, predicates_per_statement


class fol_to_fol:
    def __init__(self):
        


