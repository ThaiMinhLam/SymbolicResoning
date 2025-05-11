"""
    Utils for langchain llm systems
"""
import torch
import os
import yaml
import json
import re
import Levenshtein
import numpy as np
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True) 

from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from itertools import combinations
from dotenv import load_dotenv, dotenv_values
from icecream import ic
from nltk.stem import WordNetLemmatizer

load_dotenv()

# Preprocessing
# Preprocessing data
def clean_text(
        text,
        methods=['rmv_link', 'rmv_punc', 'lower', 'rmv_space'],
        custom_punctuation = '!"#$%&\'()*+,.-:;<=>?@[\\]^_/`{|}~”“',
    ):
    cleaned_text = text
    for method in methods:
        if method == 'rmv_link':
            # Remove link
            cleaned_text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned_text)
            cleaned_text = "".join(cleaned_text)
        elif method == 'rmv_punc':
            # Remove punctuation
            cleaned_text = re.sub('[%s]' % re.escape(custom_punctuation), '' , cleaned_text)
        elif method == 'lower':
            # Lowercase
            cleaned_text = cleaned_text.lower()
        elif method == 'rmv_space':
            # Remove extra space
            cleaned_text = re.sub(' +', ' ', cleaned_text)
            cleaned_text = cleaned_text.strip()
    return cleaned_text

# Write json
def save_json(content, save_path):
    with open(save_path, 'w') as file:
        json.dump(content, file, ensure_ascii=False, indent=4)
def save_json_append(data, filepath):
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # If file exists and is non-empty, load & append
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        with open(filepath, 'r+', encoding='utf-8') as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
            # make sure it's a list
            if not isinstance(existing, list):
                existing = [existing]
            existing.append(data)
            f.seek(0)
            json.dump(existing, f, ensure_ascii=False, indent=2)
            f.truncate()
    else:
        # First write: wrap in list
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([data], f, ensure_ascii=False, indent=2)
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
def load_finetune_model(model_base, peft_path, device):
    model = PeftModel.from_pretrained(
        model_base,
        peft_path,
        torch_dtype=torch.float16
    )
    model = model.cuda(device)    # <-- Và ép model về cuda:0 luôn
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
    generation_config.num_beam = config['num_beam']   # <-- chỗ này sửa "nun_beam" => "num_beam" luôn nếu config đúng
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
    return llm, model, tokenizer, text_pipeline



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

#-------------------------------MAP ENTITIES FROM PREDICATE NAME TO ARGUMENTS-------------------------
def get_args(predicate):
    matches = re.search(r'\(([^()]*)\)', predicate)
    if matches:
        arguments = matches.group(1)
        arguments = arguments.split(",")
        arguments = [arg.strip() for arg in arguments]
        return arguments
    return []

def check_exist_entity_in_args_old(predicate):
    arguments = get_args(predicate)
    variables = ["x", "y", "z", "t"]
    for arg in arguments:
        if arg not in variables:
            return True
    return False


def check_exist_entity_in_args_new_old(predicate):
    arguments = get_args(predicate)
    variables = ["x", "y", "z", "t"]
    for arg in arguments:
        is_variable_in_arguments = [arg in variables for arg in arguments]
        total_variable_in_arguments = sum(is_variable_in_arguments)
        if total_variable_in_arguments == 0 and len(arguments)==1: # Arg có entity - Xét trường hợp này HigherThanA(John)
            return False
        if total_variable_in_arguments == 0 and len(arguments)>1: # Arg có entity - Xét trường hợp này HigherThanA(John)
            return True
        if total_variable_in_arguments >= 2: # Đã general - HigherThan(x, y)
            return True
        if total_variable_in_arguments==1 and len(variables) > total_variable_in_arguments: # Không convert trường hợp này - HigherThan(x, John, ...)
            return True
    return False


def check_exist_entity_in_args(predicate):
    arguments = get_args(predicate)
    variables = ["x", "y", "z", "t"]
    is_variable_in_arguments = [arg in variables for arg in arguments]
    total_variable_in_arguments = sum(is_variable_in_arguments)
    if total_variable_in_arguments == 0 and len(arguments)==1: # Arg có entity - Xét trường hợp này HigherThanA(John)
        # return False - Cân nhắc
        return False
    if total_variable_in_arguments == 0 and len(arguments)>1: # Arg có entity - Ko Xét trường hợp này HigherThan(John, A)
        return True
    if total_variable_in_arguments >= 2: # Đã general - HigherThan(x, y)
        return True
    if total_variable_in_arguments==1 and len(arguments) > total_variable_in_arguments: # Không convert trường hợp này - HigherThan(x, John, ...)
        return True
    return False


def preprocessing_entity_map(old_predicate, new_predicate, list_predicate_name_cluster):
    old_predicate_name = get_name_predicate(old_predicate)
    new_predicate_name = get_name_predicate(new_predicate)
    arguments = get_args(new_predicate)
    arguments = [clean_text(arg, methods=["rmv_punc", "rmv_space"]) for arg in arguments]
    new_arguments_dict = {}
    for arg in arguments:
        new_arguments_dict[arg] = None
        if arg in ["x", "y", "z"]:
            new_arguments_dict[arg] = arg
            continue
        arg_split = arg.split(" ") # Msc Degree -> [Msc, Degree]
        arg_join = "".join(arg_split) # Msc Degree -> MscDegree
        for sub_arg in arg_split:
            common_among_cluster = [sub_arg in predicate_name_cluster for predicate_name_cluster in list_predicate_name_cluster]
            total_common_among_cluster = sum(common_among_cluster)
            if sub_arg in old_predicate_name:
                if total_common_among_cluster==0 or total_common_among_cluster>1: # Not exist in predicate_name or multiple predicate has same sub_entity
                    continue
                elif total_common_among_cluster==1: # Unique entity
                    new_arguments_dict[arg] = sub_arg
                    break
                new_arguments_dict[arg] = sub_arg
        if new_arguments_dict[arg] == None:
            new_arguments_dict[arg] = arg_join
    arguments = [new_arguments_dict[arg] for arg in arguments]
    if None in arguments:
        arguments.remove(None)
    
    final_predicate = f"{new_predicate_name}({', '.join(arguments)})"
    return final_predicate


def map_to_fol_entity(maps, fols):
    new_fols = []
    for fol in fols:
        predicates = extract_predicate_from_fol(fol)
        for predicate in predicates:
            if predicate in maps.keys():
                fol = fol.replace(predicate, maps[predicate])
        new_fols.append(fol)
    return new_fols

def map_to_lp_entity(maps, lps):
    new_lps = []
    for lp in lps:
        predicate = lp.split(":::")[0].strip()
        if predicate in maps.keys():
            lp = lp.replace(predicate, maps[predicate])
        new_lps.append(lp)
    return new_lps


def filter_duplicate_predicate(lps):
    exist_lp_predicate_name = []
    filter_lps = []
    for lp in lps:
        lp_name = lp.split(":::")[0].strip()
        if lp_name not in exist_lp_predicate_name:
            filter_lps.append(lp)
            exist_lp_predicate_name.append(lp_name)
    return filter_lps


def split_by_uppercase(s):
    return re.findall(r'[A-Z][a-z]*', s)


def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    words = split_by_uppercase(text)
    words = [lemmatizer.lemmatize(word.lower(), pos='r') for word in words]
    words = [lemmatizer.lemmatize(word.lower(), pos='n') for word in words]
    words = [lemmatizer.lemmatize(word.lower(), pos='a') for word in words]
    words = [lemmatizer.lemmatize(word.lower(), pos='v') for word in words]
    words = [word.capitalize() for word in words]
    return "".join(words)


def lemmatize_word_fol(fol):
    lemma_dic = {}
    predicates = extract_predicate_from_fol(fol)
    predicate_names = [predicate.split("(")[0] for predicate in predicates]
    for predicate, predicate_name in zip(predicates, predicate_names):
        arguments = predicate.replace(predicate_name, "")
        lemma_name = lemmatize_words(predicate_name)
        new_predicate = lemma_name + arguments
        lemma_dic[predicate] = new_predicate
    
    for k, v in lemma_dic.items():
        fol = fol.replace(k, v)
    return fol


def dummy_conversion(text1, text2):
    split_1 = split_by_uppercase(text1)
    split_2 = split_by_uppercase(text2)
    num_combination = min(len(split_1), len(split_2))

    if len(split_1) == len(split_2):
        return {}

    if len(split_1) == num_combination:
        words = split_2
        other_words = split_1
    else:
        words = split_1
        other_words = split_2

    if len(other_words) <= 2:
        return {}

    for i in range(num_combination, num_combination - 2, -1):
        words_combination = ["".join(c) for c in combinations(words, i)]
        common_word = "".join(other_words[:i])
        if common_word in words_combination:
            return {
                text1: common_word,
                text2: common_word
            }
    return {}


def create_common_map_dict(cluster):
    cluster_predicates = [lp.split(":::")[0].strip().split("(")[0] for lp in cluster]
    pair_ids = [list(pair) for pair in combinations(range(len(cluster)), 2)]

    common_list = []
    for pair in pair_ids:
        pair_predicates = np.array(cluster_predicates)[pair]
        common_word = dummy_conversion(pair_predicates[0], pair_predicates[1])
        common_list.append(common_word)
    where_exist_common = [idx for idx, predicate in enumerate(common_list) if predicate != {}]
    convert_common_dict = {k: v for dic in common_list for k, v in dic.items()}
    return convert_common_dict


def convert_common_fol(cluster, fols):
    convert_common_dict = create_common_map_dict(cluster)
    new_fols = []
    for fol in fols:
        list_predicates = extract_predicate_from_fol(fol)
        list_predicate_names = [predicate.split("(")[0] for predicate in list_predicates]
        for predicate_name in list_predicate_names:
            if predicate_name in convert_common_dict.keys():
                fol = fol.replace(predicate_name, convert_common_dict[predicate_name])
        new_fols.append(fol)
    return new_fols

def convert_common_lp(cluster):
    convert_common_dict = create_common_map_dict(cluster)
    final_lp = []
    for lp in cluster:
        lp_predicate_name = lp.split("(")[0].strip()
        if lp_predicate_name in convert_common_dict.keys():
            lp = lp.replace(lp_predicate_name, convert_common_dict[lp_predicate_name])
        ic(final_lp)
        final_lp.append(lp)
    return final_lp


#-------------------------------REDUCE_PREDICATE-------------------------
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

    distance = Levenshtein.distance(tokens1, tokens2)
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


def parse_map_predicate(full_text, cal_distance=True, threshold=0.61, select_top=None):
    redundant_predicates = {}
    lines = full_text.splitlines()
    for line in lines:
        if select_top != None and len(redundant_predicates) == select_top: # Select top records
            break
        if "replaced by" in line:
            matches = re.findall(r'(\w+\(.*?\))', line)
            if matches:
                redundants = matches[:-1]
                redundant_names = [redundant.split('(')[0] for redundant in redundants]
                general = matches[-1]
                general_name = general.split('(')[0]
                # Cal distance
                for redundant, redundant_name in zip(redundants, redundant_names):
                    if redundant in redundant_predicates.keys():
                        continue
                    # Xét trường hợp tên premise giống nhau nhưng args khác nhau
                    if redundant_name == general_name and redundant != general: # Map preidcate with same name but different parameter
                        redundant_predicates[redundant] = general
                        continue
                    # Kiểm tra similarity
                    elif cal_distance and not is_nearly_similar(redundant_name, general_name, threshold):
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

def get_name_predicate(predicate):
    return predicate.split("(")[0]

def map_to_fol(maps, logic_program, fol):
    # Variables
    maps = {
        k: v 
        for k, v in maps.items() 
        if get_name_predicate(k) not in [get_name_predicate(general) for general in maps.values() if general != v]
    } # General cannot be replaced
    logic_program = logic_program
    fols = fol
    predicates = [item.split(":::")[0].strip() for item in logic_program]

    new_fols = []
    for fol in fols:
        predicates = extract_predicate_from_fol(fol)
        for predicate in predicates:
            if predicate in maps.keys():
                fol = fol.replace(predicate, maps[predicate])
        new_fols.append(fol)
    return new_fols

#-------------------------EXTRACT_LOGIC_PROGRAM-----------------------------
def format_sep(lp: str):
    """
        Format lp seperate predicate and definition using "::" or ":" but not ":::"
    """
    if ":::" in lp:
        None
    elif "::" in lp:
        lp = lp.replace("::", ":::")
    elif ":" in lp:
        lp = lp.replace(":", ":::")
    return lp


def get_lp_info(response):
    lines_with_numbers = re.findall(r'^\d+\..*$', response, re.MULTILINE)
    predicates = [extract_predicate_from_fol(item.split(":")[0]) for item in lines_with_numbers]
    definitions = [item.split(":")[1].strip() for item in lines_with_numbers]
    lp = {predicate: definition for predicate, definition in zip(predicates, definitions)}
    return lp


def get_lp_info_v2(response):
    pairs = re.findall(r"(\w+\(.*?\)):(.*)", response)
    predicates = [pair[0].strip() for pair in pairs]
    definitions =[pair[1].strip() for pair in pairs]
    lp = {predicate: definition for predicate, definition in zip(predicates, definitions)}
    return lp


def handle_missing_predicates_with_same_name(lps, fols):
    '''
    If there are multiple predicate with same name but different arguments - Model just identify one: 
    Args:
        - lps: LogicProgram
        - fols: FOLs
        
    '''
    # Find all predicate in fols
    all_predicates = [extract_predicate_from_fol(fol) for fol in fols]
    all_predicates = np.concatenate(all_predicates).tolist()

    # Extract predicate and definition
    lps_predicates = [lp_line.split(":::")[0].strip() for lp_line in lps]
    lps_definitions = [lp_line.split(":::")[1].strip() for lp_line in lps]
    lps_names = [predicate.split("(")[0] for predicate in lps_predicates]

    # Find predicate in question not in premises
    unselected_predicates = [predicate for predicate in all_predicates if predicate not in lps_predicates]
    
    # Mapping
    for predicate in unselected_predicates:
        name = predicate.split("(")[0]
        # If name exist in the premises
        if name in lps_names:
            p_definition_idx = np.where(np.array(lps_names) == name)[0]
            try:
                p_definition = lps_definitions[p_definition_idx]
            except:
                p_definition = lps_definitions[p_definition_idx[0]]
            lps.append(f'{predicate} ::: {p_definition}')
    return lps

#---------------------------------PREPROCESSING FOL------------------------
def clean_nl(nl: str, is_multiple_choice=False):
    """
        Clean nl premise before extract fol using logicllama
        Clean nl question after make_conclusion 
    """
    # Remove A., B., C., D. (Only for multiple choice question)
    if is_multiple_choice:
        nl = re.sub(r'[A-D]\.', '', nl)

    # Replace ". " with "." (Dr. John) -> DrJohn
    nl = re.sub(r'\b([A-Z][a-z]*)\.\s+([A-Z][a-zA-Z]*)\b', r'\1\2', nl)
    return nl.strip()


def clean_fol(fol: str, is_multiple_choice=False):
    """
    Clean fol question after extract fol using logicllama
    """
    # Remove A., B., C., D. (Only for question)
    if is_multiple_choice:
        fol = re.sub(r'\b[A-D]\s', ' ', fol)
    if fol[-1] == ":":
        fol = fol[:-1]
    return fol.strip()