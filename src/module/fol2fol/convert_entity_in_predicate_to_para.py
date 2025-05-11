# Format dataset by data/train_v2_ful_fol.json records
#---------------------------------
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
from utils import load_yml, load_llm, load_json, save_json, extract_predicate_from_fol, parse_map_predicate
from src.dataloader import XAIDataset, load_dataloader 
from src.chat_agent import ChatAgent, Prompt
from src.module.reasoning import CONVERT_INDIVIDUAL_TO_PARAM
load_dotenv()


def parse_logic_program(logic_programs: list):
    '''
        Yield: predicate, natural languages
    '''
    for logic_program in logic_programs:
        pairs = logic_program.split(':::')
        predicate = pairs[0].strip()
        nl = pairs[1].strip() if len(pairs) == 2 else None
        yield predicate, nl


class ChatAgentConvertIndividual2Params(ChatAgent):
    def __init__(self, model, config):
        super().__init__(model, config)

    def make_prompt(self, lp_list):
        # PROMPT TEMPLATE
        llama2_chat_prompt_template = """
            <s>[INST] <<SYS>>
            ### Instruction:
            {instruct_prompt}

            <</SYS>>
            ### Question
            {user_question}
            Format these predicate for me.
            [/INST]
        """

        # INPUT
        #----1
        lp_prompt_template = """
            "Predicate": {predicate} ::: {nl_explain}
        """

        lp_samples = [{
            "idx": idx + 1,
            "predicate": predicate,
            "nl_explain": nl_explain
        } for idx, (predicate, nl_explain) in enumerate(parse_logic_program(lp_list))]
        
        lp_samples_obj = Prompt(
            template=lp_prompt_template,
            input_variables=["predicate", "nl_explain"]
        )
        lp_samples_obj.create_fewshot_template(
            lp_samples,
            prefix="List of Predicates and Definitions I would like to format is:")
        lp_samples_prompt = lp_samples_obj.get_prompt({})

        BACKGROUND_PROMPT = CONVERT_INDIVIDUAL_TO_PARAM()
        final_prompt_obj = Prompt(
            template=llama2_chat_prompt_template,
            input_variables=['instruct_prompt', 'user_question']
        )
        final_prompt_obj.create_prompt_template()
        final_prompt = final_prompt_obj.get_prompt({
            'instruct_prompt': BACKGROUND_PROMPT,
            'user_question': f'{lp_samples_prompt}',
        })
        return final_prompt
    

def convert_entity(model, cluster, config):
    print("Convert")
    chat_agent_convert = ChatAgentConvertIndividual2Params(model, config)  
    final_prompt = chat_agent_convert.make_prompt(
        lp_list=cluster
    )
    extract_lp_results = chat_agent_convert.inference(
        prompt=final_prompt,
        input_values={},
    )
    output = extract_lp_results['text'].split("[/INST]")[-1]
    ic(output)
    output = output.split("[/INST]")[-1]
    convert_entity_dict = parse_map_predicate(output, cal_distance=False, threshold=0)
    return convert_entity_dict
