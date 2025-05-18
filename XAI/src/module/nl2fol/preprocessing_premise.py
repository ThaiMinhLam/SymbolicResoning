import torch
from functools import partial
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel, prepare_model_for_kbit_training
import json
import time
import re
from icecream import ic

import spacy
from collections import Counter
nlp = spacy.load("en_core_web_sm")

PROMPT_LOGIC_FACT = """<s>[INST]
### Task: Rewrite a natural language premise into a logical implication form

You are given:
- A natural language premise: a sentence or statement written in plain human language (not formal logic), which expresses a fact, rule, condition, or observation about the world, a system, or a policy.
Your goal is to determine whether this premise expresses a logical implication, and if so, rewrite it as a full conditional sentence.

### **Step-by-step instructions**:
**Step 1. Analyze the Given Premise**:
- Read the premise carefully.
- Understand its **logical structure and meaning**.
- Identify any dependency or consequence relation, especially those that imply a condition leads to a result.
- Pay attention to modal verbs (e.g., "must", "should", "will", "can", "may"), causal phrases (e.g., "result in", "leads to"), and logical operators (e.g., "If", "Only if", "Unless", "Provided that", "Assuming that"), and relative clause (e.g., "Who/Which/That", "When").
- When the premise **presents a list of actions, events, entities seperated by semicolons, commas or conjunction ('and')**, **interpret them as sequential or co-occurring facts rather than inferring a conditional (if–then) relationship, unless an explicit condition is clearly stated**.

**Step 2. Attempt to Rewrite the Sentence into the Form: "If [condition], then [consequence]"**:
- If the sentence describes the **existence** of something or someone (e.g., "There exists...", "Some objects are..."), **do not rewrite it** into an "If ..., then ..." format. These sentences describe a situation rather than a conditional dependency.
- Only rewrite sentences into "If ..., then ..." form if they explicitly describe a rule, requirement, or cause-effect relationship between a condition and a consequence.
- The rewritten sentence must preserve the **exact meaning** of the original sentence.
- Do **not** add new assumptions or external knowledge.
- Ensure the rewritten form expresses a fully **logical or rule-based dependency** (i.e., something depends on something else).
- Avoid paraphrasing that weakens, changes, or generalizes the premise.
- If the premise contains grouped or itemized data (e.g., multiple scores, dates, or attributes in parentheses), ensure each item is preserved completely in the rewritten sentence.

**Step 3. Check for redundancy**:
- If the **condition and consequence** in the sentence express the **same idea** or essentially repeat each other, then **do not** convert it into an implication.
- For example, if the premise expresses a **direct equivalence** or **self-evident fact**, where both parts of the sentence convey **the same meaning** or **restate the same relationship**, then avoid writing it as an implication.
- **Do not extract or rewrite only a part of the provided premise**, especially if it leaves out important context or information. Ensure that **the entire premise, including all key details and dependencies, is fully preserved and represented** in the rewritten form.
- Ensure that **all information and the original meaning of the premise are fully preserved during the conversion without adding, omitting, or altering any details**.

**Step 4. Rewrite only implication sentences**:
- Only if the sentence expresses a true **logical implication** (a condition that leads to a result), rewrite it into the format:
  - `If [condition], then [consequence]`
- Do not **change, weaken, or generalize meaning**.
- Do not invent missing parts.
- Just use **information and word** from provided natural language premise.

**Step 5. Output requirements**:
- If valid implication:  
  `Rewritten: `"<If ..., then ...>"`
- If existential or not an implication:  
  `Rewritten: `NA`
- Do **not** add explanation, justification, or extra text. Only return the single line "Rewritten: ..."


### Example (Do **not use it in the output**, just **follow the format**):
- Example 1:
    + Natural language premise: "Some students have access to University housing."
    + Rewritten: NA
    
- Example 2:
    + Natural language premise: "Thesis completion guarantees Scholarship eligibility."
    + Rewritten: "If a student completes their thesis, then they are eligible for a scholarship."
    
### Input:
Natural language premise: {premise_NL}
[/INST]
Output: </s>"""

class Preprocessing_PremiseNL:
    def __init__(self, base_model, prompt_template_path, max_output_len, tokenizer, load_in_8bit=True):
        self.model = base_model
        self.prompt_template_path = prompt_template_path
        self.max_output_len = max_output_len
        self.load_in_8bit = load_in_8bit
        self.tokenizer = tokenizer
    
    def extract_rewritten(self, rewritten: str):
        pattern = r"Rewritten:\s*(?:\"(.*?)\"|<(.*?)>|'(.*)'|(.*))"
        matches = re.search(pattern, rewritten)
        if matches:
            return matches.group(1)
        return None

    def bool_implication(self, premise: str):
        if 'then...' in premise or 'NA' in premise:
            return False
        return True

    def extract_detail_items(self, sentence: str):
        items = {
            'token': [],
            'detail': []
        }
        pattern = re.compile(r"([\w\d\.]+)\s*\(([^)]+)\)")
        matches = pattern.findall(sentence)
        if matches:
            for match in matches:
                token, details = match
                items['token'].append(token)
                items['detail'].append(details)
        return items
    
    def remove_intro_phrase_if_common(self, sentences: list[str]):
        def extract_intro_phrase(sentence: str):
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
        
        intro_phrases = [extract_intro_phrase(sen) for sen in sentences]
        phrase_counts = Counter(intro_phrases)
        most_common_phrase, freq = phrase_counts.most_common(1)[0]
        
        if most_common_phrase and freq == len(sentences):
            return [s.replace(most_common_phrase, "").lstrip().capitalize() for s in sentences]
        else:
            return sentences
        
    def _is_valid_rewrite(self, original_items, rewritten: str) -> bool:
        new_items = self.extract_detail_items(rewritten)

        if not original_items['detail'] or not new_items['detail']:
            return True  # No details to compare

        original_sets = [set(d.split()) for d in original_items['detail']]
        new_sets = [set(d.split()) for d in new_items['detail']]

        for orig, new in zip(original_sets, new_sets):
            common = orig & new
            if len(common) != len(orig) and len(common) != len(new):
                return False
        return True
    
    def create_new_premises(self, premises: list[str]):
        new_premises = []
        fact_indices = []
        premises = self.remove_intro_phrase_if_common(premises)
        for i, premise in enumerate(premises):
            original_items = self.extract_detail_items(premise)
            prompt = PROMPT_LOGIC_FACT.format(premise_NL=premise)
            
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=self.max_output_len,
            )
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            final_result = decoded_output.split("</s>")[1]
            output = self.extract_rewritten(final_result)
            if output is None:
                new_premises.append(premise)
                fact_indices.append(i)
                continue
            
            if self._is_valid_rewrite(original_items, output) and self.bool_implication(output):
                new_premises.append(output)
            else:
                new_premises.append(premise)
                fact_indices.append(i)
                
        return new_premises, fact_indices