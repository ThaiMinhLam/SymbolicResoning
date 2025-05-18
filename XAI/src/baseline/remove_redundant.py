import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
import re
from icecream import ic
from pprint import pprint
import time

from dotenv import load_dotenv, dotenv_values 
from tqdm import tqdm

# %--- LangChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder,
)
import sys
sys.path.append("/data/npl/ViInfographicCaps/Contest/final_contest/XAI")
from utils.utils import load_yml, load_llm, save_json, parse_map_predicate, filter_similar_predicate, get_main_predicate, extract_predicate_from_fol
from src.dataloader import XAIDataset, load_dataloader 
from src.chat_agent import ChatAgent, Prompt
from src.module.reasoning import EXTRACT_MAIN_PREDICATE, HEAD_INSTRUCTION, REDUCE_AND_MATCHING_PREDICATE_PROMPT
load_dotenv()

'''
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    Matching predicate from context to question, and then extract predicate of question and trackback context 
'''

def get_args():
    parser = argparse.ArgumentParser(description="Load model config and run something")
    
    parser.add_argument('--file_path', type=str, required=True, help='Path to Reasoning Json File')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--device', type=int, required=True, default='cuda:0', help='Path to YAML config file')
    
    return parser.parse_args()


class ReasoningDataset(XAIDataset):
    

    def __init__(self, annotation_path, num_samples='all'):
        super().__init__(annotation_path, num_samples)

    def sampling(self, num_samples):
        """
            numsamples: str or int
                "all": select all
                int: select number 
        """
        samples = []
        num_records = 0
        data = self.annotation
        for id in tqdm(range(len(data))):
            item_value = data[id]
            premises = ' '.join(item_value['premises-NL'])
            fol_premises = '.'.join(item_value['premises-FOL'])
            questions = item_value['questions']
            answers = item_value['answers']
            reasonings = item_value['explanation']
            idxes = item_value['idx']
            logic_program_predicate_LLM = item_value['logic_program_predicate_LLM']
            llm_fol = item_value['LLM-FOL']

            # Create samples
            for q_id, (question, answer, reasoning, idx) in enumerate(zip(questions, answers, reasonings, idxes)):
                sub_questions = question.split(', and')
                for sub_question in sub_questions:
                    sample_item = {
                        'id': id,
                        'q_id': q_id,
                        'idx': idx,
                        'premises': premises,
                        'fol_premises': fol_premises,
                        'conclusion': sub_question.strip(),
                        'reasoning': reasoning,
                        'answer': answer, 
                        'logic_program_predicate_LLM': logic_program_predicate_LLM, 
                        'llm_fol': llm_fol, 
                    }
                    samples.append(sample_item)
                num_records += 1
            

                if num_samples != "all" and num_records >= num_samples:
                    return samples
        return samples


class ChatAgentReduce(ChatAgent):
    def __init__(self, model, config):
        super().__init__(model, config)

    def make_prompt(self, lp_predicates_list, main_predicates):
        # PROMPT TEMPLATE
        llama2_chat_prompt_template = """
            <s>[INST] <<SYS>>
            ### Instruction:
            {instruct_prompt}

            <</SYS>>
            ### Question
            {user_question} [/INST]
        """

        lp_predicates_prompt_template = """
            **Predicate**: {predicate} ::: {nl_explain}
        """

        # Logic Program example
        lp_predicates_samples = [{
            "predicate": predicate,
            "nl_explain": nl_explain
        } for predicate, nl_explain in parse_logic_program(lp_predicates_list)]
        
        # Input Context
        lp_predicates_samples_obj = Prompt(
            template=lp_predicates_prompt_template,
            input_variables=["predicate", "nl_explain"]
        )
        lp_predicates_samples_obj.create_fewshot_template(
            lp_predicates_samples,
            prefix="List of **Predicates**")
        lp_predicates_samples_prompt = lp_predicates_samples_obj.get_prompt({})

        
        # INSTRUCT PROMPT
        BACKGROUND_PROMPT = REDUCE_AND_MATCHING_PREDICATE_PROMPT()
        BACKGROUND_PROMPT = BACKGROUND_PROMPT.format(
            main_predicates=main_predicates
        )
        # FINAL PROMPT
        final_prompt_obj = Prompt(
            template=llama2_chat_prompt_template,
            input_variables=['instruct_prompt']
        )
        final_prompt_obj.create_prompt_template()
        final_prompt = final_prompt_obj.get_prompt({
            'instruct_prompt': BACKGROUND_PROMPT,
            'user_question': f'{lp_predicates_samples_prompt}',
        })
        return final_prompt

class ChatAgentFindMainPredicate(ChatAgent):
    def __init__(self, model, config):
        super().__init__(model, config)

    def make_prompt(self, list_questions):
        # PROMPT STATEMENT INPUT
            # {{ user_message_1 }} [/INST] {{ model_answer_1 }} </s>
            # <s>[INST]
        fs_item_template = """
            Statement {id}: {question}
        """

        llama2_chat_prompt_template = """
            <s>[INST] <<SYS>>
            ### Instruction:
            {instruct_prompt}

            <</SYS>>

            {user_question} [/INST]
        """

        fewshot_examples = [{
            "id": id + 1, 
            "question": question,
        } for id, question in enumerate(list_questions) if question != ""]
        statement_prompt_obj = Prompt(
            template=fs_item_template,
            input_variables=["id", "question"]
        )
        statement_prompt_obj.create_fewshot_template(
            fewshot_examples,
            prefix="")
        statement_prompt = statement_prompt_obj.get_prompt({})

        # INSTRUCTION PROMPT
        instruction_prompt_obj = Prompt(
            template=EXTRACT_MAIN_PREDICATE(),
            input_variables=["instruction"],
        )
        instruction_prompt_obj.create_prompt_template()
        instruction_prompt = instruction_prompt_obj.get_prompt({
            'instruction': HEAD_INSTRUCTION(),
        })

        # FINAL PROMPT
        final_prompt_obj = Prompt(
            template=llama2_chat_prompt_template,
            input_variables=['instruct_prompt', 'user_question']
        )
        final_prompt_obj.create_prompt_template()

        final_prompt = final_prompt_obj.get_prompt({
            'instruct_prompt': instruction_prompt, 
            'user_question': statement_prompt,
        })
        return final_prompt


def parse_logic_program(logic_programs: list):
    '''
        Yield: predicate, natural languages
    '''
    for logic_program in logic_programs:
        pairs = logic_program.split(':::')
        predicate = pairs[0].strip()
        nl = pairs[1].strip() if len(pairs) == 2 else None
        yield predicate, nl


def find_main_predicate(model, config, premises, questions):
    chat_agent = ChatAgentFindMainPredicate(model, config)
    dict_info = None
    premises = premises.split(".")
    # premises.extend(questions)
    paraphrase_prompt = chat_agent.make_prompt(premises)
    for num_regenerate in tqdm(range(2), desc=("Generate response")):
        context_results = chat_agent.inference(
            prompt=paraphrase_prompt,
            input_values={},
        )
        response = context_results['text']
        ic(response)
        dict_info = get_main_predicate(response)
        if len(dict_info['list predicates']) != 0:
            combine_predicates = []
            for item in dict_info['list predicates']:
                combine_predicates.extend(item)
            filter_list = filter_similar_predicate(combine_predicates)
            main_predicates = ", ".join([f"{item}" for item in filter_list])
            return main_predicates
    return ""

def map_to_fol(map_info_dict):
    # Variables
    maps = map_info_dict['maps']
    maps = {k: v for k, v in maps.items() if k not in maps.values()} # General cannot be replaced
    logic_program = map_info_dict['logic_program']
    fols = map_info_dict['fol']
    predicates = [item.split(":::")[0].strip() for item in logic_program]

    new_fols = []
    for fol in fols:
        predicates = extract_predicate_from_fol(fol)
        for predicate in predicates:
            if predicate in maps.keys():
                fol = fol.replace(predicate, maps[predicate])
        new_fols.append(fol)
    return new_fols


def reducing(model, reasoning_dataset, config):
    # Load dataset path
    save_dir = 'save'
    save_path = os.path.join(save_dir, 'redundant_predicates.json')
    
    print("Start Mapping")
    chat_agent_reducing = ChatAgentReduce(model, config)
    # chat_agent_reasoning_without_premises = ChatAgentReduce(model, config)
    save_dict = {}
    for i in range(len(reasoning_dataset)):
        # if i < 0:
        #     continue
        save_dict[i] = {}

        # Logic Programs
        logic_program_predicates = reasoning_dataset[i]['logic_program_predicate_LLM']
        logic_program_premises = reasoning_dataset[i]['llm_fol']
        

        lp_predicates_list = logic_program_predicates
        lp_premises_list = logic_program_premises
        
        # Input question
        ## NOTE: Question chỗ này là từng question nhỏ chứ ko phải là list các question. Vô module chính chỉnh lại thành một list các question ban đầu
        premises = reasoning_dataset[i]['premises']
        question = reasoning_dataset[i]['conclusion']
        questions = [question]
        main_predicates = find_main_predicate(model, config, premises, questions)
        ic(main_predicates)
        final_prompt = chat_agent_reducing.make_prompt(
            lp_predicates_list=lp_predicates_list,
            main_predicates=main_predicates,
        )

        # Parsing mapping
        for j in tqdm(range(3)):
            tqdm.write(f"Mapping values iter {j}")
            mapping_results = chat_agent_reducing.inference_direct(
                prompt=final_prompt,
            )
            mapping_output = mapping_results['text'].split("[/INST]")[-1]
            ic(mapping_output)
            mapping_dict = parse_map_predicate(mapping_output, 0.5)
            if len(mapping_dict) != 0:
                break
        save_dict[i]['results'] = mapping_results
        save_dict[i]['maps'] = mapping_dict
        save_dict[i]['logic_program'] = logic_program_predicates
        save_dict[i]['fol'] = logic_program_premises

        # Map to fol
        new_fols = map_to_fol(save_dict[i])
        save_dict[i]['new_fol'] = new_fols
        
        # Print results
        ic(mapping_results)

    save_json(save_dict, save_path)

if __name__=="__main__":
    begin = time.time()
    args = get_args()
    config = load_yml(args.config)
    config['file_path'] = args.file_path

    # Create model
    print("Load Model")
    model = load_llm(
        model_id=config['model_id'],
        config=config['model_config'],
        model_type=config['model_type'],
        device=args.device,
    )

    print("Load dataset")
    reasoning_dataset = ReasoningDataset(config['file_path'], num_samples='all')
    
    # Reducing
    reducing(model, reasoning_dataset, config)

    # Calculate time
    end = time.time()
    execute_time = end - begin
    ic(execute_time)
    