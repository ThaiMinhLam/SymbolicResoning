import json
import re 


def load_json(file_path):
    """
    Load a JSON file and return its content.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def extract_predicates(sample: dict):
    premises_fol = sample.get("LLM-FOL", [])
    questions_fol = sample.get("question-FOL", [])
    premises_nl = sample.get("premises-NL", [])
    questions_nl = sample.get("questions", [])

    all_nl = premises_nl + questions_nl
    all_statements = premises_fol + questions_fol

    predicates_per_statement = []

    for stmt in all_statements:
        temp = []
        pred_matches = re.findall(r'([a-zA-Z_]+)\(([^)]+)\)', stmt)
        for pred_name, args in pred_matches:
            temp.append(f"{pred_name}({args})")
        predicates_per_statement.append(temp)

    return all_nl, predicates_per_statement


data = load_json("/data/npl/ICEK/News/SymbolicResoning/data/updated_hard_samples_v3.json")
for sample in data:
    all_nl, predicates_per_statement = extract_predicates(sample)
    predicates = [item for sublist in predicates_per_statement for item in sublist]
    print(predicates)
    break


