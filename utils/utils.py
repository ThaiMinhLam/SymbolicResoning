"""
    Utils for langchain llm systems
"""
import torch
import os
import yaml
import json
import re
import Levenshtein

from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from dotenv import load_dotenv, dotenv_values
from icecream import ic
load_dotenv()

# Trackkkk
# Write json
def save_json(content, save_path):
    with open(save_path, 'w') as file:
        json.dump(content, file, ensure_ascii=False, indent=4)

# Load json
def load_json(path):
    with open(path, 'r') as file:
        content = json.load(file)
        return content

# Load yml
def load_yml(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return None


# Load LLM
# https://www.mlexpert.io/blog/langchain-quickstart-with-llama-2
def load_llm_cuong(model_id, config, model_type="", device='cuda'):
    token = os.getenv(f"HF_TOKEN_{model_type.upper()}")
    if token == None:
        assert Exception("No HF_TOKEN for MODEL_TYPE founded")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": '<unk>',
        "pad_token": '<unk>',
    })  
    
    generation_config = GenerationConfig.from_pretrained(model_id)
    generation_config.temperature = config['temperature']
    generation_config.top_p = config['top_p']
    generation_config.top_k = config['top_k']
    generation_config.num_beams = config['num_beams']


def load_finetune_model(model_base, peft_path, device):
    model = PeftModel.from_pretrained(
        model_base,
        peft_path,
        torch_dtype=torch.float16
    )
    model = model.to(f'cuda:{device}')    # <-- Và ép model về cuda:0 luôn
    return model


def load_llm(model_id, config, model_type="llama", device='cuda'):
    token = os.getenv(f"HF_TOKEN_{model_type.upper()}")
    if token is None:
        raise Exception("No HF_TOKEN for MODEL_TYPE found")  # assert phải bỏ đi, dùng raise

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    tokenizer.padding_side = "left"  # BẮT BUỘC vì model là decoder-only (Llama, GPT, v.v.)
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "<unk>",
    })

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        token=token
    )

    # Load generation config
    generation_config = GenerationConfig.from_pretrained(model_id)
    generation_config.max_new_tokens = config['max_new_tokens']
    generation_config.temperature = config['temperature']
    generation_config.top_p = config['top_p']
    generation_config.do_sample = config['do_sample']
    generation_config.repetition_penalty = config['repetition_penalty']
    generation_config.num_beam = config['nun_beam']   # <-- chỗ này sửa "nun_beam" => "num_beam" luôn nếu config đúng
    generation_config.dola_layers = config['dola_layers']
    generation_config.use_cache = config['use_cache']

    # Build pipeline
    text_pipeline = pipeline(
        config['task'],
        model=model.cuda(device),
        tokenizer=tokenizer,
        generation_config=generation_config,
        device=device
    )

    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": config['temperature']})
    return llm, model, tokenizer



# Postprocessing
def get_paraphrase_info(response):
    text = response.split("<</SYS>>")[-1]
    pattern_statements = r"Simplified Statement \d+: .*$"
    # pattern_objects = r"List Objects \d+: .*$"
    # pattern_actions = r"List Actions \d+: .*$"
    pattern_predicates = r"List Predicates \d+: .*$"
    pattern_instances = r"List Instances \d+: .*$"
    pattern_information = r"Important Information \d+: .*$"

    matches_statements = re.findall(pattern_statements, text, re.MULTILINE)
    # matches_objects = re.findall(pattern_objects, text, re.MULTILINE)
    # matches_actions = re.findall(pattern_actions, text, re.MULTILINE)
    matches_predicates = re.findall(pattern_predicates, text, re.MULTILINE)
    matches_instances = re.findall(pattern_instances, text, re.MULTILINE)
    matches_information = re.findall(pattern_information, text, re.MULTILINE)
    
    # return matches_statements, matches_objects, matches_actions, matches_instances, matches_information
    return matches_statements, matches_predicates, matches_instances, matches_information


def get_matching_info(response):
    text = response.split("<</SYS>>")[-1]
    pattern_matches = r"Matching \d+: .*$"
 
    matches_statements = re.findall(pattern_matches, text, re.MULTILINE)
   
    # return matches_statements, matches_objects, matches_actions, matches_instances, matches_information
    return matches_statements



def parse_info(info, sep=','):
    '''
        Sep to split into array
    '''
    match = re.search(r':\s*(.*)', info).group(1)
    if sep != None:
        items = match.split(sep)
        return items
    return match

#-------------------------------REDUCE_PREDICATE----------------------
# Reduce Predicate
def longest_common_substring(str1, str2):
    rows = len(str1) + 1
    cols = len(str2) + 1
    length_matrix = [[0 for _ in range(cols)] for _ in range(rows)]

    max_length = 0
    end_index = 0 
    for i in range(1, rows):
        for j in range(1, cols):
            if str1[i - 1] == str2[j - 1]:
                length_matrix[i][j] = length_matrix[i - 1][j - 1] + 1
                if length_matrix[i][j] > max_length:
                    max_length = length_matrix[i][j]
                    end_index = i

    longest_substring = str1[end_index - max_length:end_index]

    return longest_substring

def is_nearly_similar(phrase1, phrase2, threshold):
    tokens1 = re.findall(r'\b\w+\b', phrase1.lower())
    tokens2 = re.findall(r'\b\w+\b', phrase2.lower())

    distance = Levenshtein.distance(phrase1, phrase2)
    max_len = max(len(phrase1), len(phrase2))
    similarity_score = 1 - (distance / max_len)

    common_substring_ratio = len(longest_common_substring(phrase1, phrase2)) / max_len
    combined_similarity = (similarity_score + common_substring_ratio) / 2
    ic(phrase1, phrase2, combined_similarity)
    return combined_similarity >= threshold and combined_similarity < 1

def parse_map_predicate_old(full_text, threshold):
    pattern = r'Predicate "([^"]+)" is redundant and can be replaced by Predicate "([^"]+)"'
    redundant_predicates = {}
    for line in full_text.split('\n'):
        match = re.search(pattern, line)
        if match:
            redundant = match.group(1)
            redundant_name = redundant.split('(')[0]
            general = match.group(2)
            general_name = general.split('(')[0]

            # Cal distance
            if not is_nearly_similar(redundant_name, general_name, threshold):
                continue
            redundant_predicates[redundant] = general
    return redundant_predicates


def parse_map_predicate(full_text, threshold):
    redundant_predicates = {}
    lines = full_text.splitlines()
    for line in lines:
        if "replaced by" in line:
            matches = re.findall(r'(\w+\(.*?\))', line)
            print(matches)
            if matches:
                redundants = matches[:-1]
                redundant_names = [redundant.split('(')[0] for redundant in redundants]
                general = matches[-1]
                general_name = general.split('(')[0]

                # Cal distance
                for redundant, redundant_name in zip(redundants, redundant_names): 
                    if not is_nearly_similar(redundant_name, general_name, threshold):
                        continue
                    redundant_predicates[redundant] = general
    return redundant_predicates


# Find main predicate
def filter_similar_predicate(all_predicates_input):
    all_predicates = all_predicates_input
    similar_predicate = []
    for i in range(len(all_predicates)):
        for j in range(i+1, len(all_predicates)):
            # similar, score = is_nearly_similar(all_predicates[i], all_predicates[j], 0.7)
            score = Levenshtein.ratio(all_predicates[i], all_predicates[j])
            if score>0.87 and score<0.99:
                sim_pair = (all_predicates[i], all_predicates[j])
                similar_predicate.append(sim_pair)
    if len(similar_predicate) == 0:
        return all_predicates
    else:
        remove_predicates = [pair[1] for pair in similar_predicate]
        for rm_pre in remove_predicates:
            if rm_pre in all_predicates:
                all_predicates.remove(rm_pre)
        return filter_similar_predicate(all_predicates)

def parse_info_predicate(info, sep=','):
    '''
        Sep to split into array
    '''
    match = re.search(r':\s*(.*)', info).group(1)
    if sep != None:
        items = match.split(sep)
        items = [item.strip() for item in items]
        return items
    return match

def get_main_predicate(response):
    text = response.split("<</SYS>>")[-1]
    pattern_statements = r"Simplified Statement \d+: .*$"
    pattern_predicates = r"List Predicates \d+: .*$"

    matches_statements = re.findall(pattern_statements, text, re.MULTILINE)
    matches_predicates = re.findall(pattern_predicates, text, re.MULTILINE)

    category_names = ['simplified statement', 'list predicates']
    categories = [matches_statements, matches_predicates]

    dic_info = {}
    for cat_name, cat in zip (category_names, categories):
        parsed_cat = [parse_info_predicate(cat_content, ',') for cat_content in cat]
        dic_info[cat_name] = parsed_cat
    return dic_info


def extract_predicate_from_fol(fol):
    matches = re.findall(r'(\w+\(.*?\))', fol)
    return matches


