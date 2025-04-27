import torch
from functools import partial
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, prepare_model_for_kbit_training
from LogicLLaMA.utils import TranslationDataPreparer
from generatev2 import llama_batch_generate
import json
import time
import re
import json
from tqdm import tqdm

def has_abcd_pattern(s: str) -> bool:
    """
    Returns True if `s` contains, in order, on separate lines:
      - a line starting with "A" 
      - then a line starting with "B"
      - then a line starting with "C"
      - then a line starting with "D"
    """
    # 
    # Explanation of the pattern:
    #  \nA[^\n]*     – a newline + “A” + anything up to the next newline
    #  \nB[^\n]*     – then newline + “B” + anything up to its newline
    #  \nC[^\n]*     – likewise for “C”
    #  \nD[^\n]*:    – then newline + “D” + anything, ending with a colon
    #
    pattern = r"\nA[^\n]*\nB[^\n]*\nC[^\n]*\nD[^\n]*"
    return bool(re.search(pattern, s))
def split_question_options(s: str):
    # Capture groups:
    # 1: question (lazy up to the line before A)
    # 2: text after "A"
    # 3: text after "B"
    # 4: text after "C"
    # 5: text after "D" (colon is matched but not included)
    capture = (
        r"^(.*?)\r?\n"       # 1: question (anything up to first newline before A)
        r"A\s*([^\n]*)\r?\n"  # 2: A-line content
        r"B\s*([^\n]*)\r?\n"  # 3: B-line content
        r"C\s*([^\n]*)\r?\n"  # 4: C-line content
        r"D\s*([^\n]*)"      # 5: D-line content (colon out of capture)
    )
    m = re.search(capture, s, flags=re.DOTALL)
    if not m:
        raise ValueError("Failed to parse question/options despite matching the pattern")

    question = m.group(1).strip()
    opts = [m.group(i).strip() for i in range(2, 6)]
    return [question, opts[0], opts[1], opts[2], opts[3]]
def combine_question_options(parts):
    """
    Given a list of exactly five strings:
      [question, optionA, optionB, optionC, optionD]
    returns a single string formatted as:

      question
      A optionA
      B optionB
      C optionC
      D optionD
    """
    q, a, b, c, d = parts
    return "\n".join([
        q.strip(),
        f"A {a.strip()}",
        f"B {b.strip()}",
        f"C {c.strip()}",
        f"D {d.strip()}:"
    ])

class nl_to_fol:
    def __init__(self, base_path, prompt_template_path, peft_path, max_output_len, load_in_8bit=True):
        self.model_path = base_path
        self.prompt_template_path = prompt_template_path
        self.load_in_8bit = load_in_8bit
        self.max_output_len = max_output_len

        self.tokenizer = self.load_tokenizer(base_path)
        self.model = self.load_model(base_path, peft_path)
        self.generation_config = self.get_generation_config()
        

    def load_tokenizer(self, base_model):
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        tokenizer.padding_side = "left"# Allow batched inference
        tokenizer.add_special_tokens({
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": '<unk>',
            "pad_token": '<unk>',
        }) 
        return tokenizer
    
    def get_generation_config(self):
        generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=1
        )
        return generation_config

    def load_model(self, base_model, peft_path):
        llama_model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=self.load_in_8bit,
            torch_dtype=torch.float16,
            device_map='auto',
        )
        llama_model = prepare_model_for_kbit_training(llama_model)


        model = PeftModel.from_pretrained(
            llama_model,
            peft_path,
            torch_dtype=torch.float16
        )
        model.to('cuda')
        return model

    def data_preparer(self):
        data_preparer = TranslationDataPreparer(
            self.prompt_template_path,
            self.tokenizer,
            False,
            256 # just a filler number
        )

        prepare_input = partial(
            data_preparer.prepare_input,
            **{"nl_key": "NL"},
            add_eos_token=False,
            eval_mode=True,
            return_tensors='pt'
        )
        batch_simple_generate = partial(
            llama_batch_generate,
            llama_model=self.model,
            data_preparer=data_preparer,
            max_new_tokens=self.max_output_len,
            generation_config=self.generation_config,
            prepare_input=prepare_input,
            return_tensors=False
        )
        return batch_simple_generate

    def generate(self, input_json, output_json):
        batch_simple_generate = self.data_preparer()
        # Set your starting index here
        start_idx = 0 

        # Load your data
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        output_path = output_json

        # Only loop from the specified index
        for idx in tqdm(range(start_idx, len(data)), desc="Processing samples"):
            sample = data[idx]

            # Extract and prepare input data
            llm_fol = []
            data_list = []
            multiple_choices_list = []
            premises = sample.get("premises-NL", [])
            for premise in premises:
                data_list.append({'NL': premise})
            sep_idx = len(data_list)

            questions = sample.get("questions", [])
            # Remove old question and replace with quest - 4 option:
            for i, question in enumerate(questions):
                if has_abcd_pattern(question):
                    multiple_choices_list.append(i)
            for i, question in enumerate(questions):
                if has_abcd_pattern(question):
                    parts = split_question_options(question)
                    questions[i:i+1]=parts

            for question in questions:
                data_list.append({'NL': question})

            # Generate output
            full_resp_str, resp_parts = batch_simple_generate(input_str=data_list)

            for fol_part in resp_parts:
                llm_fol.append(fol_part[1])

            # Retry for any `None` values
            none_indices = [i for i, pair in enumerate(resp_parts) if any(elem is None for elem in pair)]
            while len(none_indices) != 0:
                print(f"GOT NONE: {none_indices}")
                retry_data = []
                for i in none_indices:
                    try:
                        retry_data.append({'NL': premises[i]})
                    except:
                        retry_data.append({'NL': questions[i - sep_idx]})

                _, retry_parts = batch_simple_generate(input_str=retry_data)
                for i, fol_part in enumerate(retry_parts):
                    llm_fol[none_indices[i]] = fol_part[1]

                none_indices = [i for i, pair in enumerate(retry_parts) if any(elem is None for elem in pair)]
            sample['LLM-FOL'] = llm_fol[:sep_idx]
            ques_fol = llm_fol[sep_idx:]
            for ques_id in multiple_choices_list:
                ques_fol[ques_id:ques_id+5] = [combine_question_options(ques_fol[ques_id:ques_id+5])]
                questions[ques_id:ques_id+5] = [combine_question_options(questions[ques_id:ques_id+5])]
            sample['question-FOL'] = ques_fol

            ##### KHÚC NÀY SAMPLE ĐÃ HOÀN CHỈNH RỒI NHA:

            
            # Save progress after each sample
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    

def main(): # pipeline
    base_model='/data/npl/ViInfographicCaps/Contest/demo_contest/xai/Llama-2-7b-chat-hf'
    peft_path='/data/npl/ICEK/LLaMA/LogicLLaMA-7b-direct-translate-delta-v0.1'
    prompt_template_path='/data/npl/ICEK/News/SymbolicResoning/prompt_templates'
    load_in_8bit = True
    max_output_len = 256
    input_json = '/data/npl/ICEK/News/SymbolicResoning/data/train_v2.json'
    output_json = '/data/npl/ICEK/News/SymbolicResoning/data/demo_v2.json'

    nl_to_fol_instance = nl_to_fol(base_model, prompt_template_path, peft_path, max_output_len, load_in_8bit)
    nl_to_fol_instance.generate(input_json, output_json)

if __name__ == "__main__":
    main()
    




