import torch
import numpy as np
from functools import partial
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel, prepare_model_for_kbit_training
import json
import time
import re
import huggingface_hub
from transformers import pipeline
from icecream import ic

PROMPT_LP = """<s>[INST]
### Task: For each given **First-Order Logic (FOL) predicate**, generate a clear and accurate **natural language definition** that faithfully captures its intended meaning based on the given natural language passage.

You are given:
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
   - The number of output lines must **exactly match** the number of FOL predicates.
   - Each definition must correspond to **only one** FOL predicate.
   - No summaries, no combined definitions, and no extra commentary.

### Input:
- List of FOL Predicates: {input_predicates}
[/INST]
Output: </s>"""

PROMPT_LP_FINAL = """<s>[INST]
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
- **The natural language definition must not include or repeat the original FOL predicate name; it must be written entirely in natural language**.

### Instructions:
1. **Define each FOL predicate as a complete, well-formed natural language sentence**:
     - **Does not require an external passage — rely only on the meaning inferred from the predicate name and its argument structure**.
     - **Analyze the predicate carefully** to understand its logical meaning and the roles of its arguments.
     - Use the **predicate name and argument structure** to determine the most natural and complete way to express its meaning in English.
     - Write a **single, grammatically complete sentence** that fully defines the predicate.
     - Clearly describe what the predicate name refers to — especially if it involves a specific category.
     - Ensure that the sentence **explicitly describes the roles of all arguments** (e.g., who performs an action, who receives it).
       
2. **Rules for constructing the definition**:
    - The definition **must be a complete natural language sentence with a clear subject and verb (predicate) structure**.
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
   - Each natural language definition must correspond to **only one** FOL predicate.
   - No summaries, no combined definitions, and no extra commentary.
    - The number of output lines must **exactly match** the number of FOL predicates.

### Input:
- Natural language Premise: {premise_nl}
- List of FOL Predicates: {input_predicates}
[/INST]
Output: </s>"""


def extract_PredicatesIndividuals(sample: dict, mode='premise') -> list:
    if mode == 'premise':
        statements_nl = sample.get("premises-nl", [])
        statements_fol = sample.get("LLM-FOL", [])
    else:
        statements_nl = sample.get("questions", [])
        statements_fol = sample.get("question-FOL", [])
        # Preprocess
        statements_nl = [nl.split('\n')[1:] if len(nl.split('\n')) > 2 else [nl] for nl in statements_nl]
        statements_nl = [nl for statement_nl in statements_nl for nl in statement_nl]
        statements_fol = [fol.split('\n')[1:] if len(fol.split('\n')) > 2 else [fol] for fol in statements_fol]
        statements_fol = [fol for statement_fol in statements_fol for fol in statement_fol ]
        
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

class Extract_Logic_Progam:
    def __init__(self, base_model, prompt_template_path, max_output_len, tokenizer, load_in_8bit=True):
        self.model = base_model
        self.prompt_template_path = prompt_template_path
        self.max_output_len = max_output_len
        self.load_in_8bit = load_in_8bit
        self.tokenizer = tokenizer

    def generate_sample(self, sample, mode):
        all_nl, predicates_entities = extract_PredicatesIndividuals(sample, mode)
        print(f"The number of predicates: {len(predicates_entities)}")
        prompt = PROMPT_LP_FINAL.format(premise_nl=all_nl, input_predicates=predicates_entities)

        # Tokenize input text
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)

        # Generate output
        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=2048,
        )

        # Decode output
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Split and postprocess
        final_result = decoded_output.split("</s>")[1]
        output = extract_result(final_result)
        return output