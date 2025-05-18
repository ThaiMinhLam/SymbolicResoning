import os, torch
from icecream import ic
from typing import Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    pipeline,
)
import re
from typing import List
os.environ["HF_TOKEN"] = "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"  
import spacy
from collections import Counter
nlp = spacy.load("en_core_web_sm")

def extract_intro_phrase(sentence):
    doc = nlp(sentence)

    comma_idx = None
    for token in doc:
        if token.text == ',':
            comma_idx = token.i
            break
    
    if comma_idx is None:
        return None
        
    for token in doc[:comma_idx]:
        if token.dep_ in ['prep', 'advmod', 'npadvmod']:
            return sentence[:doc[comma_idx + 1].idx].strip()
    
    return None

def remove_intro_phrase_if_common(sentences):
    intro_phrases = [extract_intro_phrase(sen) for sen in sentences]
    phrase_counts = Counter(intro_phrases)
    most_common_phrase, freq = phrase_counts.most_common(1)[0]
    
    if most_common_phrase and freq == len(sentences):
        return [s.replace(most_common_phrase, "").lstrip().capitalize() for s in sentences]
    else:
        return sentences

def extract_PredicatesIndividuals(sample: dict) -> list:
    statements_nl = sample.get("premises-NL", [])
    statements_fol = sample.get("LLM-FOL", [])

    predicates_entities = set()
    
    for i_th, fol in enumerate(statements_fol):
        if fol is None:
            continue
        # Find predicate names
        pred_matches = re.findall(r'([a-zA-Z0-9_]+)\(([^)]+)\)', fol)
        for pred_name, args in pred_matches:
            pred_name = re.sub('¬', '', pred_name)
            predicate = f"{pred_name}({args})"
            predicates_entities.add(predicate)
    return statements_nl, list(predicates_entities)

def extract_result(output_text):
    output = []
    pattern = r"[a-zA-Z0-9_]+\(.*?\) \:+ (.*)"
    lines = output_text.split("\n")
    for text in lines:
        match = re.search(pattern, text)
        if match:
            start = match.start()
            end = match.end()
            output.append(text[start:end].strip())
    return output


def clean_predicate(phrase):
    return remove_regex.sub('', phrase.strip())

def split_compound(phrase):
    return [p.strip() for p in re.split(r'\band\b|\bor\b', phrase)]

def extract_clean_predicates(raw_phrase):
    predicates = []
    for part in split_compound(raw_phrase):
        cleaned = clean_predicate(part)
        if cleaned:
            predicates.append(cleaned)
    return predicates

def generate_fol(pipeline, prompt: str):
    ic("generate_fol")
    outputs = pipeline(prompt)
    return outputs[0]["generated_text"]

def generate_predicate_fol(pipeline, prompt: str):
    ic("generate_predicate_fol")
    outputs = pipeline(prompt)
    return outputs[0]["generated_text"]

    
def template_create_FOL(premises_nl: str, logic_program: List[str], subject: list[str]):
    prompt = f"""
    [INST]
    You are a formal-logic expert.
    Your task: convert each natural-language statement (Premise-NL)
    into one first-order-logic formula (Premise-FOL) in *logic-program style*.

    ╭─ OUTPUT RULES (mandatory) ─────────────────────────────────────────╮
    │ • Return **one line per premise**, nothing else.                  │
    │ • Line = pure FOL: no bullet numbers, no “:::”, no code-fences.   │
    │ • Keep antecedent ⇒ consequent order exactly as in NL.            │
    │ • Allowed symbols:  ∀  ∃ / Exists  →  ∧  ∨  ¬                    │
    ╰────────────────────────────────────────────────────────────────────╯

    GENERAL CONVENTIONS  
    1. **Predicates** (section ❶) are declared as  
        <Predicate>(<vars>) ::: <NL description>  
    → *Never* insert constants there; always use a variable (x, y…).  

    2. **Arity guide** — when to choose 1-place vs 2-place  
    • `P(x)`  for an intrinsic property or an action toward a fixed object.  
    • `R(x,y)` for relations where y can vary / be queried later.  
    • If unsure, prefer 2-place; you can always constrain later.  

    3. **Subject handling**  
    • *Class name* (Student, FacultyMember…): add that predicate in the
        antecedent → `FacultyMember(x) → …`.  
    • *Generic word* (person, someone, individual…): omit any `Person(x)`
        unless NL states it.

    4. **Named-individual rule**  
    • If NL refers to a **specific proper name** (e.g. “Professor John”),
        treat it as a **constant**  `John`.  
    • Do **not** quantify over a constant; simply use it in the formula:  
        `CompletedTraining(John) → CanTeachUndergrad(John)`  

    5. Named-individual fact  
    • If a clause simply asserts **one property** of a specific, named
        individual (John, Course101…), output **exactly one
        predicate(constant)**—no quantifier, no arrow, no condition.  
        NL: “Professor John holds a PhD.”  
        FOL: HoldsPhD(John)

    WORKFLOW FOR EACH SENTENCE  
    S1  Detect quantifier pattern: universal, existential, none (named individual).  
    S2  Map NL phrases to declared predicates; apply arity & subject rules.  
    S3  Build the formula, adding ∀x / ∃x only when the sentence is about
        unspecified individuals.  
    S4  Output the single-line formula.

    [/INST]
    [INPUT]
    "premises_nl": {premises_nl}
    "subject": {subject}
    "logic_program": {logic_program}
    [/INPUT]
    [OUTPUT]
    """
    return prompt

def extract_FOL_from_llm(raw: str):
    
    return raw.split('[OUTPUT]')[-1].strip()

def template_create_FOL_predicate(predicate_nl: List[str])
    return f"""<s>[INST]
    ### Task: For each given **natural language predicate**, generate a corresponding **First-Order Logic (FOL) predicate** that faithfully captures its logical meaning.

    You are given:
    - A **premise** that provides the contextual background or knowledge base from which the logical predicates are derived.
    - Each sentence describes a logical relationship or property involving one or more entities.
    - Each definition is a **complete English sentence** and clearly specifies the **roles of all arguments** (e.g., who performs an action, who receives it).

    #### Your Goal:
    For every natural language predicate, produce a well-formed **FOL predicate expression** that:
    - Accurately reflects the **logical meaning** conveyed by the sentence.
    - Identifies the **correct number of arguments** (e.g., x, y, z) based on the entities involved.
    - Creates a **clear and self-explanatory predicate name** (in English) that summarizes the core relationship or property described.
    - Assigns each argument to its correct role (e.g., subject, object, action recipient).
    - Uses **logical arguments (e.g., x, y, z)** to represent the entities involved.
    - Reflects the **correct argument structure** (e.g., number of arguments, their logical roles).
    - Uses logical notation: `PredicateName(x)`, `PredicateName(x, y)`, etc.

    ### Instructions:
    1. **Analyze each natural language predicate carefully** to determine:
        - The number of **entities** involved (i.e., how many arguments the predicate should take).
        - The **core relationship or property** being described (used to form the predicate name).
        - The **roles** of the arguments (e.g., agent, theme, location, object).
        - If the predicate involves a **fixed object like a degree, title, award, job, etc.**, represent it as a **constant**, not part of the predicate name.

    2. **Rules for constructing the FOL predicate**:
        - The **predicate name** must summarize the core action/relation/property using meaningful English terms.
        - The number of arguments in the predicate must match the number of entities involved.
        - Maintain the **semantic roles** of the arguments
        - Avoid copying the sentence structure directly — instead, generate a compact and expressive predicate name.
        - Use consistent naming style: either `CamelCase` or `snake_case`.
        - **Do NOT merely paraphrase or copy** the sentence — you must **analyze** and convert it into a **logical representation**.

    3. **Output Format**:
        - Each output must be exactly **one line per definition**.
        - Each line must follow this format:
            Natural Language Predicate ::: FOL-Predicate
        - Use **exactly three colons** (` ::: `) as the separator.
        - Example:
            + is a Teacher ::: Teacher(x)
            + do test on y subject ::: DoTestSubject(x, y)

    4. **Final Requirements**:
        - You must define **every single natural language predicate** provided in the list.
        - Each FOL predicate must be **logically sound, syntactically correct, and match the right number of arguments and roles**.
        - **Do not skip any natural language predicates**.
        - No extra commentary or explanation — only the list of converted predicates.
        - The number of output lines must **exactly match** the number of input definitions.

    ### Input:
    - List of Natural Language Predicate: {predicate_nl}
    [/INST]
    Output: </s>"""





if __name__ == "__main__":
    import time
    start_time = time.time()
    premises_NL =[
            "If a faculty member has completed training, they can teach undergraduate courses.",
            "If a faculty member can teach undergraduate courses and holds a PhD, they can supervise graduate students.",
            "If a faculty member can supervise graduate students and has at least 3 publications, they can serve on curriculum committees.",
            "If a faculty member can serve on curriculum committees and has a positive teaching evaluation, they can propose new courses.",
            "Professor John has completed pedagogical training.",
            "Professor John holds a PhD.",
            "Professor John has published at least 3 academic papers.",
            "Professor John has received a positive teaching evaluation."
        ]

    lp_sub = lp_sub = [
    {
        "subject": "faculty member",
        "logic_program": [
            "CompletedTraining(x) ::: has completed training",
            "CanTeachUndergrad(x) ::: can teach undergraduate courses"
        ]
    },
    {
        "subject": "faculty member",
        "logic_program": [
            "CanTeachUndergrad(x) ::: can teach undergraduate courses",
            "HoldsPhD(x) ::: holds a PhD",
            "CanSuperviseGradStudents(x) ::: can supervise graduate students"
        ]
    },
    {
        "subject": "faculty member",
        "logic_program": [
            "CanSuperviseGradStudents(x) ::: can supervise graduate students",
            "HasAtLeast3Publications(x) ::: has at least 3 publications",
            "CanServeOnCommittees(x) ::: can serve on curriculum committees"
        ]
    },
    {
        "subject": "faculty member",
        "logic_program": [
            "CanServeOnCommittees(x) ::: can serve on curriculum committees",
            "HasPositiveTeachingEval(x) ::: has a positive teaching evaluation",
            "CanProposeNewCourses(x) ::: can propose new courses"
        ]
    },
    {
        "subject": "Professor John",
        "logic_program": [
            "CompletedTraining(John) ::: has completed pedagogical training"
        ]
    },
    {
        "subject": "Professor John",
        "logic_program": [
            "HoldsPhD(John) ::: holds a PhD"
        ]
    },
    {
        "subject": "Professor John",
        "logic_program": [
            "HasAtLeast3Publications(John) ::: has published at least 3 academic papers"
        ]
    },
    {
        "subject": "Professor John",
        "logic_program": [
                "HasPositiveTeachingEval(John) ::: has received a positive teaching evaluation"
        ]
    }
    ]

    logic_program = ["PythonCode(x) ::: If a Python code is well-tested, then the project is optimized.",
            "WellTested(x) ::: If a Python code does not follow PEP 8 standards, then it is not well-tested.",
            "OptimizedProject(x) ::: All Python projects are easy to maintain.",
            "PEP8Standards(x) ::: If a Python code follows PEP 8 standards, then it is easy to maintain.",
            "PythonProject(x) ::: If a Python code is well-tested, then it follows PEP 8 standards.",
            "EasyToMaintain(x) ::: If a Python project is well-structured, then it is optimized.",
            "FollowsPEP8Standards(x) ::: If a Python code is well-tested, then it follows PEP 8 standards.",
            "WellTestedPythonCode(x) ::: If a Python project is well-structured, then it has clean and readable code.",
            "WellStructured(x) ::: If a Python project is well-tested, then it has clean and readable code.",
            "Optimized(x) ::: If a Python project is easy to maintain, then it is optimized.",
            "CleanCode(x) ::: If a Python code is well-tested, then it has clean and readable code.",
            "ReadableCode(x) ::: If a Python project is well-structured, then it has clean and readable code.",
            "Person(y) ::: There exists at least one Python project that follows best practices.",
            "Perfect(y) ::: There exists at least one Python project that is optimized.",
            "FollowsBestPractices(x) ::: If a Python project is not well-structured, then it does not follow PEP 8 standards.",
            "Clean(x) ::: If a Python project is not well-structured, then it does not follow PEP 8 standards.",
            "Readable(x) ::: If a Python project is not well-structured, then it does not follow PEP 8 standards."]
    
    premises_FOL = [
        "∀x (WT(x) → O(x))",
      "∀x (¬PEP8(x) → ¬WT(x))",
      "∀x (EM(x))",
      "∀x (WT(x))",
      "∀x (PEP8(x) → EM(x))",
      "∀x (WT(x) → PEP8(x))",
      "∀x (WS(x) → O(x))",
      "∀x (EM(x) → WT(x))",
      "∀x (O(x) -> CR(x))",
      "∀x (WS(x))",
      "∀x (CR(x))",
      "∃x (BP(x))",
      "∃x (O(x))",
      "∀x (¬WS(x) → ¬PEP8(x))"
    ]

    idx = [1]
    question = "Which conclusion follows with the fewest premises?\nA. If a Python project is not optimized, then it is not well-tested\nB. If all Python projects are optimized, then all Python projects are well-structured\nC. If a Python project is well-tested, then it must be clean and readable\nD. If a Python project is not optimized, then it does not follow PEP 8 standards"
    answer =  "A" 
    for i in range(len(premises_NL)):
        subject = lp_sub[i]['subject']
        logic_prog = lp_sub[i]['logic_program']
        premise = premises_NL[i]

        prompt = template_create_FOL(premise, logic_prog, subject)
        output = generate_fol(pipeline, prompt)
        print(f"[{i+1}] Premise-NL: {premise}")
        print(f"FOL Output:\n{output.split('[OUTPUT]')[-1].strip()}\n")
    end_time = time.time()
    print(end_time - start_time)