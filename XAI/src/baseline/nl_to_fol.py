import sys
sys.path.append("/data/npl/ICEK/News/SymbolicResoning")
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
def has_comma_and_pattern(s: str) -> bool:
    """
    Returns True if `s` contains the substring ", and" anywhere.
    """
    # simple regex for a comma + space + "and"
    pattern = r", and"
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
def split_double_question(parts):
    return parts.split(", and")
def combine_double_question(parts):
    return "<q>".join(parts)
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

def retry_fill(fol_list, data_list, generate_fn):
    """
    Repeatedly call `generate_fn` on any positions where fol_list[i] is None,
    pulling the same NL inputs from data_list until no slots remain None.
    """
    none_idxs = [i for i, v in enumerate(fol_list) if v is None]
    while none_idxs:
        print(f"GOT NONE at positions: {none_idxs}")
        retry_input = [data_list[i] for i in none_idxs]
        _, retry_parts = generate_fn(input_str=retry_input)
        # retry_parts is a list of (inp_dict, fol_str)
        for orig_idx, (_, new_fol) in zip(none_idxs, retry_parts):
            fol_list[orig_idx] = new_fol
        none_idxs = [i for i, v in enumerate(fol_list) if v is None]
    return fol_list


class nl_to_fol:
    def __init__(self, base_model, prompt_template_path, peft_path, max_output_len, load_in_8bit=True, is_pipeline=False):
        self.is_pipeline = is_pipeline
        self.model_base = base_model
        self.prompt_template_path = prompt_template_path
        self.load_in_8bit = load_in_8bit
        self.max_output_len = max_output_len

        if self.is_pipeline:
            # Nếu truyền vào đã là pipeline thì không load lại model/tokenizer
            self.tokenizer = base_model.pipeline.tokenizer
            self.model = base_model.pipeline.model
            self.generation_config = base_model.pipeline.generation_config
        else:
            self.tokenizer = self.load_tokenizer(base_model)
            self.model = self.load_model(peft_path)
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

    def load_model(self, peft_path):
        # llama = LlamaForCausalLM.from_pretrained(
        #     base_model,
        #     torch_dtype=torch.float16,
        #     device_map={"": 0},  
        # )

        model = PeftModel.from_pretrained(
            self.model_base,
            peft_path,
            torch_dtype=torch.float16
        )
        model = model.to('cuda:0')    # <-- Và ép model về cuda:0 luôn
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

    def generate_dataset(self, input_json, output_json):
        batch_simple_generate = self.data_preparer()
        # Set your starting index here
        start_idx = 0 

        # Load your data
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        output_path = output_json

        # Only loop from the specified index
        # Only loop from the specified index
        for idx in tqdm(range(start_idx, len(data)), desc="Processing samples"):
            sample = data[idx]

            premises = sample.get("premises-NL", [])
            raw_questions = sample.get("questions", [])

            # 1) Flatten questions, record where we expanded MCQ vs “, and”
            flat_qs = []
            mcq_positions = []      # list of (start_idx, option_count)
            comma_and_positions = []  # list of start_idx

            for q in raw_questions:
                if has_abcd_pattern(q):
                    parts = split_question_options(q)
                    mcq_positions.append((len(flat_qs), len(parts)))
                    flat_qs.extend(parts)
                elif has_comma_and_pattern(q):
                    parts = split_double_question(q)
                    comma_and_positions.append(len(flat_qs))
                    flat_qs.extend(parts)
                else:
                    flat_qs.append(q)

            # 2) Build data_list in one go
            data_list = (
                [{"NL": p} for p in premises] +
                [{"NL": q} for q in flat_qs]
            )
            sep_idx = len(premises)

            # 3) Generate and retry‐fill
            full_str, resp_parts = batch_simple_generate(input_str=data_list)
            llm_fol = [fol for _, fol in resp_parts]
            llm_fol = retry_fill(llm_fol, data_list, batch_simple_generate)

            # 4) Slice out premises vs question‐FOL
            sample['LLM-FOL'] = llm_fol[:sep_idx]
            ques_fol = llm_fol[sep_idx:]

            # 5) Combine back in **reverse** order so earlier splices don't shift later ones
            for start, count in sorted(mcq_positions, reverse=True):
                slice_ = ques_fol[start:start+count]
                merged = combine_question_options(slice_)
                ques_fol[start:start+count] = [merged]

            for start in sorted(comma_and_positions, reverse=True):
                slice_ = ques_fol[start:start+2]
                merged = combine_double_question(slice_)
                ques_fol[start:start+2] = [merged]

            sample['question-FOL'] = ques_fol

            # Save progress after each sample
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)


    
    def generate_sample(self, sample):
        batch_simple_generate = self.data_preparer()

        premises = sample.get("premises-nl", [])
        raw_questions = sample.get("questions", [])

        # 1) Flatten questions, record where we expanded MCQ vs “, and”
        flat_qs = []
        mcq_positions = []      # list of (start_idx, option_count)
        comma_and_positions = []  # list of start_idx

        for q in raw_questions:
            if has_abcd_pattern(q):
                parts = split_question_options(q)
                mcq_positions.append((len(flat_qs), len(parts)))
                flat_qs.extend(parts)
            elif has_comma_and_pattern(q):
                parts = split_double_question(q)
                comma_and_positions.append(len(flat_qs))
                flat_qs.extend(parts)
            else:
                flat_qs.append(q)

        # 2) Build data_list in one go
        data_list = (
            [{"NL": p} for p in premises] +
            [{"NL": q} for q in flat_qs]
        )
        sep_idx = len(premises)

        # 3) Generate and retry‐fill
        full_str, resp_parts = batch_simple_generate(input_str=data_list)
        llm_fol = [fol for _, fol in resp_parts]
        llm_fol = retry_fill(llm_fol, data_list, batch_simple_generate)

        # 4) Slice out premises vs question‐FOL
        sample['LLM-FOL'] = llm_fol[:sep_idx]
        ques_fol = llm_fol[sep_idx:]

        # 5) Combine back in **reverse** order so earlier splices don't shift later ones
        for start, count in sorted(mcq_positions, reverse=True):
            slice_ = ques_fol[start:start+count]
            merged = combine_question_options(slice_)
            ques_fol[start:start+count] = [merged]

        for start in sorted(comma_and_positions, reverse=True):
            slice_ = ques_fol[start:start+2]
            merged = combine_double_question(slice_)
            ques_fol[start:start+2] = [merged]

        sample['question-FOL'] = ques_fol


        return sample

# def main(): # pipeline
#     base_model='/data/npl/ViInfographicCaps/Contest/demo_contest/xai/Llama-2-7b-chat-hf'
#     peft_path='/data/npl/ICEK/LLaMA/LogicLLaMA-7b-direct-translate-delta-v0.1'
#     prompt_template_path='/data/npl/ICEK/News/SymbolicResoning/prompt_templates'
#     load_in_8bit = True
#     max_output_len = 256
#     input_json = '/data/npl/ICEK/News/SymbolicResoning/data/train_v2.json'
#     output_json = '/data/npl/ICEK/News/SymbolicResoning/data/demo_v2.json'

#     nl_to_fol_instance = nl_to_fol(base_model, prompt_template_path, peft_path, max_output_len, load_in_8bit)
#     nl_to_fol_instance.generate_dataset(input_json, output_json)

# if __name__ == "__main__":
#     main()
    




