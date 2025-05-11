import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
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
from utils import load_yml, load_llm, load_json, save_json, parse_info, get_matching_info
from src.dataloader import XAIDataset, load_dataloader 
from src.chat_agent import ChatAgent, Prompt
from src.module.lina import PARAPHRASE_PROMPT_TEMPLATE, MATCHING_PROMPT_TEMPLATE, HEAD_INSTRUCTION
load_dotenv()

'''
    Matching predicate from context to question, and then extract predicate of question and trackback context 
       
'''

def get_args():
    parser = argparse.ArgumentParser(description="Load model config and run something")
    
    parser.add_argument('--paraphrase', type=str, required=True, help='Path to Paraphrase Json File')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--device', type=int, required=True, default='cuda:0', help='Path to YAML config file')
    
    return parser.parse_args()


class ChatAgentMatchingStatement(ChatAgent):
    def __init__(self, model, config):
        super().__init__(model, config)

    def make_prompt(self, statements, predicates, instances):
        # PROMPT STATEMENT INPUT
            # {{ user_message_1 }} [/INST] {{ model_answer_1 }} </s>
            # <s>[INST]

        llama2_chat_prompt_template = """
            <s>[INST] <<SYS>>
            ### Instruction:
            {instruct_prompt}

            <</SYS>>
            ### Question
            {user_question} [/INST]
        """

        fs_item_template = """
            Statement {id}: {statement}
            List Predicates {id}: {predicate}
            List Instances {id}: {instances}
        """

        fewshot_examples = [{
            "id": id + 1, 
            "statement": statement,
            "predicate": ", ".join(predicate),
            "instance": ", ".join(instance),
        } for id, (statement, predicate, instance) in enumerate(zip(statements, predicates, instances))]

        statement_prompt_obj = Prompt(
            template=fs_item_template,
            input_variables=["id", "statement", "predicate"]
        )
        statement_prompt_obj.create_fewshot_template(
            fewshot_examples,
            prefix="")
        statement_prompt = statement_prompt_obj.get_prompt({})
        
        # INSTRUCTION PROMPT
        instruction_prompt_obj = Prompt(
            template=MATCHING_PROMPT_TEMPLATE(),
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


def postpreprocessing(response):
    matches_statements = get_matching_info(response=response)
    ic(matches_statements)
    category_names = ["matching_info"]
    categories = [matches_statements]

    dic_info = {}
    for cat_name, cat in zip (category_names, categories):
        parsed_cat = [parse_info(cat_content, ",") for cat_content in cat]
        dic_info[cat_name] = parsed_cat
    return dic_info


def matching(config, device):
    # Load Examples path
    save_dir = 'save'
    save_path = os.path.join(save_dir, 'matching.json')
    examples_file = load_json(config['paraphrase_json'])
    # ic((examples_file.values()))
    statements_collection = [item['statement'] for item in examples_file.values()]
    questions_collection = [item['questions'] for item in examples_file.values()]
    predicates_collection = [item['list predicates'] for item in examples_file.values()]
    instances_collection = [item['list instances'] for item in examples_file.values()]
    
    # Load ChatAgent
    model = load_llm(
        model_id=config['model_id'],
        config=config['model_config'],
        model_type=config['model_type'],
        device=device,
    )

    print("Chain example")
    
    chat_agent = ChatAgentMatchingStatement(model, config)
    matching_content = {}
    dict_info = {}
    for idx, (statements, questions, predicates, instances) in enumerate(zip(statements_collection, questions_collection, instances_collection)):
        matching_prompt = chat_agent.make_prompt(statements, predicates, instances)
        for num_regenerate in tqdm(range(5), desc=("Matching Context")):
            matching_results = chat_agent.inference(
                prompt=matching_prompt,
                input_values={},
            )
            response = matching_results['text']
            ic(response.split("### Answer Response:")[-1])
            # info = postpreprocessing(response)
            dict_info[num_regenerate] = response
            ic(dict_info)
            # if len(info['matching_info']) != 0:
            #     break
        matching_content[idx] = dict_info     
    save_json(matching_content, save_path)
    # ic(context_results.split('$$$$')[-1])


if __name__=="__main__":
    begin = time.time()
    args = get_args()
    config = load_yml(args.config)
    config['paraphrase_json'] = args.paraphrase
    matching(config, args.device)
    end = time.time()
    execute_time = end - begin
    ic(execute_time)
