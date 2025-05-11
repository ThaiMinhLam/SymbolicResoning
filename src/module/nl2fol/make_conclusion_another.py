import torch
from functools import partial
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel, prepare_model_for_kbit_training
import json
import time
import re
from icecream import ic


PROMPT_CREATE_HYPOTHESIS_ANOTHER = """<s>[INST]
### Task: 
Your task is to transform this question into a single declarative sentence called a hypothesis, which expresses the core meaning or assumption of the question in statement form.

You are given:
- A Natural language Question: It may be in the form of WH-questions (e.g., What, Why, How, When, Where), Yes/No/Uncertain question, or other interrogative forms. The question typically contains a focus or target of inquiry and can often be rephrased into a statement (hypothesis) to be evaluated based on relevant information.
- This question typically seeks information or clarification and may be rephrased into a hypothesis to be evaluated as true, false, or uncertain given certain context (premises).

### What is a Hypothesis?
A hypothesis is a **single declarative sentence** that expresses what would be true **if the answer to the question is "Yes"**. It must:
- Be **grammatically complete**.
- Be **logically faithful** to the question's meaning.
- Be **evaluatable** as true, false, or uncertain given external context or premises (not included in this prompt).
- **Avoid including phrases** like “according to the premises”, “based on the context”, “as per the passage”, etc., as these are implied.

- Example for Yes/No question type and its hypothesis:
  + Question: "Does Sophia qualify for the university scholarship, according to the premises?"
  + Hypothesis: Sophia qualify for the university scholarship.

### Output Requirements:
- The output **must start with** `Hypothesis:` followed by exactly **one declarative sentence**.
- The hypothesis must:
  - Be **grammatically complete**.
  - Be a **faithful and concise restatement** of the question’s intent.
  - Be evaluable as **true**, **false**, or **uncertain** based on supporting context.
- Do **not** return a question.
- Do **not** include explanations, reasoning, or invented content.
- Do **not** use markdown, bullet points, or formatting symbols.
- The output must be a **single line** only.

### Input:
Natural language question: {question_NL}
[/INST]
Output: </s>"""

class Extract_Hypothesis:
    def __init__(self, base_model, prompt_template_path, max_output_len, tokenizer, load_in_8bit=True):
        self.model = base_model
        self.prompt_template_path = prompt_template_path
        self.max_output_len = max_output_len
        self.load_in_8bit = load_in_8bit
        self.tokenizer = tokenizer
    
    def generate_hypothesis(self, question):
        _match = re.search(r"Statement:\s*['\"]?([^'\"]+)['\"]?", question)
        if _match:
            hypothesis = _match.group(1)
        else:
            # Setup Prompt
            prompt = PROMPT_CREATE_HYPOTHESIS_ANOTHER.format(question_NL=question)
            
            # Tokenize input text
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)
            
            # Generate output
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=self.max_output_len,
            )
            
            # Decode output
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Split and postprocess
            final_result = decoded_output.split("</s>")[1]
            match1 = re.search(r"Hypothesis:\s*(.*)", final_result)
            if match1:
                hypothesis = match1.group(1)
            else:
                hypothesis = final_result
                
        return hypothesis