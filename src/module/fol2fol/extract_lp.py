import torch
from functools import partial
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel, prepare_model_for_kbit_training
import json
import time
import re
import huggingface_hub
import nltk
from nltk.stem import WordNetLemmatizer

TOKEN_READ_HF = "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"
TRAINING_DATA = "train_v3.json"
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

PROMPT = """<s>[INST]
### Task: Define the meaning of each FOL predicate individually by directly extracting from the corresponding natural language (NL) statement.
You are given:
- A list of Natural Language (NL) statements, where each statement describes a context, domain, or situation.
- A corresponding list of lists of First-Order Logic (FOL) predicates extracted from those contexts.

Please follow these instructions carefully:
1. Interpret the NL statement: 
  - Understand the general context and concepts described.
  
2. Define each FOL predicate:
  - For each predicate in the list at position i:
    - Carefully read the corresponding NL statement.
    - Identify the smallest possible phrase (or fragment) directly from the NL statement that fully captures the intended meaning of the predicate:
       - The fragment must be short, precise, and taken from the NL wording.
       - It must accurately and completely reflect the meaning required by the predicate.
    - Check the selected phrase by answering::
       (a) Does it fully cover the intended meaning of the predicate?
       (b) Is it directly quoted or minimally adapted from the NL statement without adding or omitting information?
       (c) Is the definition short, clear, and faithful to the wording and semantics of the NL statement?
    - Only after verifying all conditions (a), (b), and (c) are satisfied, output the final definition.
    
3. Use the required output format rules:
  - For each predicate to be defined, output must strictly follow this exact structure for each line: FOL-Predicate ::: Natural Language Definition
    - Exactly three colons (" ::: ") between the FOL predicate and its definition.
    - Examples for each predicate and its definition: 
      + Teacher(x) ::: x is a Teacher
      + DoTestSubject(x, y) ::: student x do test on y subject
    - Do not include examples and explanation in the output.
    - Only one line output per predicate.

**Additional guidelines**:
 - Keep each definition **concise, accurate, and faithful** to the given NL statement.
 - Only describe what each predicate represents.
 - The number of output lines must exactly match the number of predicates given.
 - Ensure all FOL predicate in the listed defined.

### Input:
- List of NL Statements: {input_statement}
- List of FOL Predicates: {input_predicates}
[/INST]
Output: </s>"""

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def exist_prev(text1, prev_list):
    for text in prev_list:
        if text1.lower() == text.lower():
            return True
    return False

def extract_PredicatesIndividuals(sample: dict) -> list:
    premises_nl = sample.get("premises-NL", [])
    premises_fol = sample.get("LLM-FOL", [])

    all_nl = premises_nl
    all_fol = premises_fol

    predicates_entities = []
    final_nl = []
    prev_entities = {}
    
    for i_th, fol in enumerate(all_fol):
        if fol is None:
            continue
        if '\n' in fol:
            fol = fol.split('\n')[0]
        # Find predicate names
        temp = []
        pred_matches = re.findall(r'([a-zA-Z_]+)\(([^)]+)\)', fol)
        for pred_name, args in pred_matches:
            predicate = f"{pred_name}({args})"
            pred_name = lemmatizer.lemmatize(pred_name.lower()).capitalize()
            pred_name = re.sub('¬', '', pred_name)
            if not exist_prev(pred_name, prev_entities.keys()):
                temp.append(predicate)
                prev_entities[pred_name] = []
            elif len(args.split(",")) not in prev_entities[pred_name]:
                temp.append(predicate)
                
            prev_entities[pred_name].append(len(args.split(",")))
            
        if temp:
            final_nl.append(all_nl[i_th])
            for pred in temp:
                predicates_entities.append(pred)
            
    return final_nl, predicates_entities

def extract_result(output_text):
    output = []
    pattern = r"[a-zA-Z]+\(.*?\) \:\:\: (.*)"
    lines = output_text.split("\n")
    for text in lines:
        match = re.search(pattern, text)
        if match:
            start = match.start()
            end = match.end()
            output.append(text[start:end].strip())
    return output

def write_file_json(file_name, dict_output):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(dict_output, f, ensure_ascii=False, indent=4)

class Extract_Logic_Progam:
    def __init__(self, base_model, prompt_template_path, max_output_len, tokenizer, load_in_8bit=True):
        self.model = self.remove_adapter(base_model)
        self.prompt_template_path = prompt_template_path
        self.max_output_len = max_output_len
        self.load_in_8bit = load_in_8bit
        self.tokenizer = tokenizer

    def remove_adapter(self, model):
        model = model.merge_and_unload()
        return model

    def extract_PredicatesIndividuals(sample: dict) -> list:
        premises_nl = sample.get("premises-nl", [])
        premises_fol = sample.get("LLM-FOL", [])

        all_nl = premises_nl
        all_fol = premises_fol

        predicates_entities = []
        final_nl = []
        prev_entities = {}
        
        for i_th, fol in enumerate(all_fol):
            if fol is None:
                continue
            if '\n' in fol:
                fol = fol.split('\n')[0]
            # Find predicate names
            temp = []
            pred_matches = re.findall(r'([a-zA-Z_]+)\(([^)]+)\)', fol)
            for pred_name, args in pred_matches:
                predicate = f"{pred_name}({args})"
                pred_name = lemmatizer.lemmatize(pred_name.lower()).capitalize()
                pred_name = re.sub('¬', '', pred_name)
                if not exist_prev(pred_name, prev_entities.keys()):
                    temp.append(predicate)
                    prev_entities[pred_name] = []
                elif len(args.split(",")) not in prev_entities[pred_name]:
                    temp.append(predicate)
                    
                prev_entities[pred_name].append(len(args.split(",")))
                
            if temp:
                final_nl.append(all_nl[i_th])
                for pred in temp:
                    predicates_entities.append(pred)
                
        return final_nl, predicates_entities


    def generate_sample(self, sample):
        all_nl, predicates_entities = self.extract_PredicatesIndividuals(sample)
        result = self.model(PROMPT.format(input_statement=all_nl, input_predicates=predicates_entities))
        final_result = result[0]['generated_text'].split("</s>")[1]
        output = extract_result(final_result)
        return output


# if __name__ == '__main__':
#     huggingface_hub.login(token=TOKEN_READ_HF)
#     base_model='meta-llama/Llama-2-7b-chat-hf'
#     prompt_template_path='data/prompt_templates'
#     load_in_8bit = True
        
#     tokenizer = LlamaTokenizer.from_pretrained(base_model)
#     tokenizer.add_special_tokens({
#         "eos_token": "</s>",
#         "bos_token": "<s>",
#         "unk_token": "<unk>",
#         "pad_token": "<unk>",
#     })
#     tokenizer.padding_side = "left"

#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",              # nf4: tốt hơn fp4
#         bnb_4bit_use_double_quant=True,         # giúp nén tốt hơn
#         bnb_4bit_compute_dtype=torch.bfloat16   # A100 hỗ trợ tốt
#     )

#     llama_model = LlamaForCausalLM.from_pretrained(
#         base_model,
#         quantization_config=bnb_config,
#         torch_dtype=torch.bfloat16,           # ưu tiên bfloat16 thay vì float16
#         device_map="auto",                    # tự chia lên multi-GPU nếu có
#         low_cpu_mem_usage=True,
#         trust_remote_code=True,               # nếu dùng repo custom
#     )

#     llama_model = prepare_model_for_kbit_training(llama_model)
#     peft_path='yuan-yang/LogicLLaMA-7b-direct-translate-delta-v0'
#     model = PeftModel.from_pretrained(
#         llama_model,
#         peft_path,
#         torch_dtype=torch.float16
#     )
    
#     # Tắt Adapter để thực hiện prompting
#     merged_model = model.merge_and_unload()
    
#     # Load Training Dataset
#     dataset = read_json(TRAINING_DATA)
    
#     # Build Pipeline
#     pipe = pipeline(task="text-generation", model=merged_model, tokenizer=tokenizer, max_length=2048)
    
#     # Run Model
#     final_output = []
#     for i_th, sample in enumerate(dataset):
#         _dict = {}
#         all_nl, predicates_entities = extract_PredicatesIndividuals(sample)
#         result = pipe(PROMPT.format(input_statement = all_nl, input_predicates=predicates_entities))
#         final_result = result[0]['generated_text'].split("</s>")[1]
#         _dict["logic_program_predicate_LLM"] = extract_result(final_result)
#         final_output.append(_dict)
    
#     # Write file json
#     write_file_json("output_lp.json", final_output)