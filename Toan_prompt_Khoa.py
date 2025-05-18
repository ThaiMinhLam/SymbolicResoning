from typing import List
import ast
# ¬
import os, torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    pipeline,
)

os.environ["HF_TOKEN"] = "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"  

MODEL_NAME = "/workspace/model_hf/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,      
    device_map="auto",          
    use_auth_token=True,
)

gen_cfg = GenerationConfig(
    task               = "text-generation",
    max_new_tokens     = 1024,
    do_sample          = True,
    temperature        = 0.1,
    top_p              = 0.95,
    repetition_penalty = 1.15,
    num_beams          = 2,        
    use_cache          = True,
    
)

mistral = pipeline(
    "text-generation",
    model      = model,
    tokenizer  = tokenizer,
    generation_config = gen_cfg,
)

def get_response(prompt:str):
    outputs = mistral(
        prompt,
        
    )
    return outputs[0]["generated_text"]

def create_predicate_fol(predicate):
    prompt = f"""
You are a symbolic logic assistant specializing in converting natural language predicates (NL) into First-Order Logic (FOL) representations.

### Task:
Convert the given natural language predicate (NL) into its corresponding FOL representation, following the rules below.

### Rules:
1. Use variable `x` for the subject (the entity being described).
2. Use variable `y` only if the predicate explicitly describes a relation between two distinct entities (e.g., "x is a friend of y").
3. If the predicate describes a property, condition, or event involving a single entity (e.g., "x is eligible"), use only `x`.
4. If the predicate describes a relation between two entities (e.g., "x is enrolled in y"), use both `x` and `y`.
5. Create a concise and meaningful predicate name:
   - Use lowercase with underscores for multi-word predicates (e.g., `is_eligible`, `enrolled_in`).
   - Reflect the meaning of the NL predicate accurately (e.g., "is eligible for graduation" → `is_eligible_for_graduation`).
6. Do not include quantifiers (e.g., ForAll, Exists) or logical operators (e.g., →, ∧) unless explicitly stated in the predicate.
7. Handle verb tenses and prepositions by incorporating them into the predicate name (e.g., "receives a scholarship from" → `receives_scholarship_from`).

### Output Format:
natural language predicate ::: FOL predicate

### Examples:
- is eligible for graduation ::: is_eligible_for_graduation(x)
- is the advisor of ::: is_advisor_of(x, y)
- receives a scholarship from ::: receives_scholarship_from(x, y)
- is enrolled in course ::: is_enrolled_in(x, y)
- has completed training ::: has_completed_training(x)

### Input:
{predicate} :::

### Instructions:
1. Determine if the predicate describes a property (1 entity) or a relation (2 entities).
2. Assign variables: use `x` for a single entity, or `x` and `y` for a relation.
3. Create a concise FOL predicate name that captures the meaning of the NL predicate.
4. Format the output as shown in the examples.
5. Double-check that the FOL representation matches the NL predicate's meaning and follows the variable rules.

Ensure the FOL representation is accurate, concise, and uses the correct variables (`x` or `x` and `y`).
"""
    return prompt
        


def get_predicate_from_nl(list_of_premises_nl):
    prompt = f"""<s>[INST]
### Task: From a list of **natural language premises**, extract **complete predicate phrases** (main actions or properties) that capture the core meaning of each sentence.

You are given:
- A list of natural language premises, each describing facts, rules, or knowledge in a domain.
- Your task is to extract **predicate phrases** from each premise — these are the main actions or properties (including passive constructions) expressed in each sentence.

---

### Extraction Rules (Apply to each premise separately):

1. **Extract only the core predicate phrase**:
   - Focus on **main actions or properties** (typically verbs and their complements/modifiers).
   - **Exclude subjects** (e.g., "students", "people") and only extract the **predicate phrase**.
   - Do **not** remove essential complements, prepositions, or passive markers.
     - ✅ Correct: "submitted before the deadline", "signed by the teacher"
     - ❌ Incorrect: "submitted", "signed"

2. **Remove auxiliary and modal verbs**:
   - Delete: am/is/are/was/were/be/do/does/did, including “not”
   - Delete: must/should/shall/will/would/can/could/may/might/ought to
   - Keep **only the lexical/main verb** and its **associated phrase**.

3. **Preserve prepositional and passive structures**:
   - Retain any parts that complete the verb’s meaning (e.g., "evaluated by the committee").

4. **Split compound predicates**:
   - If multiple predicates are joined by "and"/"or", **split them** into separate predicate phrases.
   - When a modifier applies to multiple predicates, **replicate it** for each.

5. **Handle conditional or negative constructions carefully**:
   - Extract the **core action separately** from conditions.
   - Do not merge conditional clauses unless they affect the **semantic meaning** of the predicate.

---

### Output Format:
- Return a **single list** of predicate phrases extracted from **all premises**.
- Use this exact format:  
  **List of predicate: ["predicate phrase 1", "predicate phrase 2", ..., "predicate phrase N"]**
- Do **not repeat** the premises or provide any explanation.
- **Only output this list**, and ensure it contains **no duplicates** unless a phrase appears with distinct meaning.

---

### Input:
- List of natural language premises:
{list_of_premises_nl}
[/INST]
Output: </s>"""
    return prompt

def premise_to_fol_prompt(logic_program, premise_nl):
        """  
        Convert a natural language premise to FOL using predefined predicates.  

        Args:  
            nl_premise (str): Natural language premise (e.g., "All cats love fish").  
            logic_program (list): List of predicates (e.g., ["cat(x) ::: x is a cat", ...]).  

        Returns:  
            str: FOL formula (e.g., "∀x (cat(x) → love(x, fish))").  
        """  
        # Construct the prompt  
        prompt = f"""  
        Convert this NL premise to FOL:  
        NL Premise: "{premise_nl}"  
        Available Predicates:  
        {chr(10).join(logic_program)}  

        Follow these rules:  
        1. Use ONLY the predicates above.  
        2. Use quantifiers (∀, ∃) and logic operators (→, ∧, ∨, ¬) as needed.  
        3. Output ONLY the FOL formula from the available predicates, no explanations.  

        FOL Formula:  
        """  

        return prompt
# Lâm bảo viết ngược lại là ra 
def predicate_NL2FOL():
    return """<s>[INST]
    ### Task: For each given **First-Order Logic (FOL) predicate**, generate a clear and accurate **natural language definition** that faithfully captures its intended meaning based on the given natural language passage.

    You are given:
    - The premise provides the contextual background or knowledge base from which the logical predicates are derived.
    - A list of **FOL predicates** that represent logical relations or properties in the domain of discourse.
    - Each predicate has a name that encodes its meaning in English (or composed of meaningful English tokens).
    - Each predicate takes a specific number of arguments (e.g., x, y), which represent entities involved in that relationship.

    #### Your Goal:
    For every predicate, write **a full, grammatically complete English sentence** that:
    - Clearly reflects the **intended logical meaning** of the predicate name and its arguments.
    - Specifies the **role** of each argument in the relationship (e.g., who performs an action, who receives it).
    - Is a **grammatically correct and complete sentence** with a clear subject and verb.
    - **Does not require an external passage** — rely only on the meaning inferred from the predicate name and its argument structure.

    ### Instructions:
    1. **Define each FOL predicate as a complete, well-formed natural language sentence**:
        - **Analyze the predicate carefully** to understand its logical meaning and the roles of its arguments.
        - Use the **predicate name and argument structure** to determine the most natural and complete way to express its meaning in English.
        - Write a **single, grammatically complete sentence** that fully defines the predicate.
        - Clearly describe what the predicate name refers to — especially if it involves a specific category.
        - Ensure that the sentence **explicitly describes the roles of all arguments** (e.g., who performs an action, who receives it).
        
    2. **Rules for constructing the definition**:
        - The definition must be a complete sentence with a clear subject and verb (predicate) structure.
        - The definition must **not be vague or overly general**.
        - Do not **modify, rename, or infer** beyond the given predicate name.
        - **Do not merge multiple predicates into one definition**. Each predicate must be defined separately.

    3. Output Format:
        - Each definition must be **exactly one sentence long**.
        - You must output **one line for each predicate**.
        - Each line must follow **exactly** this format:
            FOL-Predicate ::: Natural Language Definition
        - Use **exactly three colons** (` ::: `) as the separator.
        - Example:
        + Teacher(x) ::: x is a Teacher
        + DoTestSubject(x, y) ::: student x do test on y subject

    4. Final Requirements:
    - You must define **every single FOL predicate** provided in the List of FOL Predicates.
    - **Do not skip the last predicate.**
    - Each definition must correspond to **only one** FOL predicate.
    - No summaries, no combined definitions, and no extra commentary.
    - You are not allowed to output fewer or more lines than the total number of predicates.

    ### Input:
    - Natural language Premise: {premise_nl}
    - List of FOL Predicates: {input_predicates}
    [/INST]
    Output: </s>"""


import re

# Danh sách các từ cần loại bỏ nếu đứng đầu
AUX_MODALS = [
    "am", "is", "are", "was", "were", "be", "do", "does", "did",
    "must", "should", "shall", "will", "would", "can", "could", "may", "might", "ought to"
]

NEGATIONS = [
    "am not", "is not", "isn't", "are not", "aren't", "was not", "wasn't", "were not", "weren't",
    "do not", "don't", "does not", "doesn't", "did not", "didn't",
    "must not", "mustn't", "should not", "shouldn't", "shall not", "shan't",
    "will not", "won't", "would not", "wouldn't", "cannot", "can't",
    "could not", "couldn't", "may not", "might not", "ought not to"
]

# Gộp hai danh sách lại để xử lý chung
REMOVE_PATTERNS = NEGATIONS + AUX_MODALS

# Tạo regex để nhận diện nếu chuỗi bắt đầu bằng các mẫu trên
remove_regex = re.compile(rf"^({'|'.join(re.escape(p) for p in sorted(REMOVE_PATTERNS, key=len, reverse=True))})\s+", re.IGNORECASE)

examples = [
    "should not be ignored by students",
    "must complete the form",
    "did not finish the report",
    "can access the database",
    "are studying at the library",
    "should and must submit the assignment"
]

def clean_predicate(phrase):
    return remove_regex.sub('', phrase.strip())

# Tách nếu có "and"/"or"
def split_compound(phrase):
    return [p.strip() for p in re.split(r'\band\b|\bor\b', phrase)]

# Tổng xử lý
def extract_clean_predicates(raw_phrase):
    predicates = []
    for part in split_compound(raw_phrase):
        cleaned = clean_predicate(part)
        if cleaned:
            predicates.append(cleaned)
    return predicates[0]

# Áp dụng thử
# for raw in premises:
#     print("Input:", raw)
#     print("List of predicate:", extract_clean_predicates(raw))
#     print()

if __name__ == "__main__":
    premise_NL = [
            "If a Python code is not well-tested, then it is not optimized.",
            "All Python projects are easy to maintain.",
            "All Python code follows PEP 8 standards.",
            "If a Python project is easy to maintain, then it is well-tested.",
            "There exists at least one Python project that is well-tested.",
            "All Python code is well-tested.",
            "If a Python code does not follow PEP 8 standards, then it is not well-tested.",
            "If a Python project is not optimized, then it is not well-tested.",
            "If a Python code is easy to maintain, then it follows PEP 8 standards.",
            "If a Python code is well-tested, then it follows PEP 8 standards.",
            "If a Python project is easy to maintain, then it is well-tested.",
            "All Python projects have clean and readable code.",
            "If a Python project is well-tested, then it is optimized."
    ]
    prompt = get_predicate_from_nl(premise_NL)
    result_get_predicate_from_nl_str = get_response(prompt).split("Output: </s>")[1].strip()
    result_get_predicate_from_nl_list = ast.literal_eval(result_get_predicate_from_nl_str)

    processed_predicate_from_nl = []
    for i in result_get_predicate_from_nl_list:
        processed_predicate_from_nl.append(extract_clean_predicates(i))

    result_get_predicate_from_nl = list(set(processed_predicate_from_nl))
    
    prompt_create_fol = create_predicate_fol(result_get_predicate_from_nl)
    print(get_response(prompt_create_fol))