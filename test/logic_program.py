import re
import json

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data



def extract_PredicatesIndividuals(sample: dict) -> list:

    premises_nl = sample.get("premises-NL", [])
    questions_nl = sample.get("questions", [])
    premises_fol = sample.get("LLM-FOL", [])
    questions_fol = sample.get("question-FOL", [])

    all_nl = premises_nl + questions_nl
    # Find all predicate names and individuals (constants)
    all_statements = premises_fol + questions_fol

    predicates_entities = []
    
    for stmt in all_statements:
        # Find predicate names
        temp = []
        pred_matches = re.findall(r'([a-zA-Z_]+)\(([^)]+)\)', stmt)
        for pred_name, args in pred_matches:
            temp.append(f"{pred_name}({args})")
        predicates_entities.append(temp)

    return all_nl, list(predicates_entities)


