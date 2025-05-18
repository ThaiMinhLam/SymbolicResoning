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
sys.path.append("/data/npl/ViInfographicCaps/Contest/final_contest/final_code")
from utils import load_yml, save_json, extract_predicate_from_fol, get_lp_info, get_lp_info_v2
from src.dataloader import XAIDataset, load_dataloader 
from src.chat_agent import ChatAgent, Prompt
from src.module.fol2fol.template import LOGIC_PROGRAM_EXTRACTION_PROMPTING_DEFINITION
load_dotenv()

'''
    Matching predicate from context to question, and then extract predicate of question and trackback context 
       
'''


class ChatAgentExtractLogicProgram(ChatAgent):
    def __init__(self, model, config):
        super().__init__(model, config)

    def make_prompt(self, premises, list_predicates_premise, list_predicates_question):
        # PROMPT TEMPLATE
        llama2_chat_prompt_template = """
            <s>[INST] <<SYS>>
            ### Instruction:
            {instruct_prompt}

            <</SYS>>
            ### Question
            {user_question} [/INST]
        """

        # INPUT
        #----1
        pair_predicate_premise_template = """
            **Predicates {id}**: {predicates}
            **Statement {id}**: {premise} 
        """
        pair_predicate_premise_samples = [{
            "id": id + 1, 
            "predicates": ", ".join(predicates),
            "premise": premise,
        } for id, (premise, predicates) in enumerate(zip(premises, list_predicates_premise))]
        pair_predicate_premise_obj = Prompt(
            template=pair_predicate_premise_template,
            input_variables=["id", "question"]
        )
        pair_predicate_premise_obj.create_fewshot_template(
            pair_predicate_premise_samples,
            prefix=""
        )
        pair_predicate_premise_prompt = pair_predicate_premise_obj.get_prompt({})

        #----2
        pair_predicate_question_template = """
            **Question-Predicates {id}:**: {q_predicates}
        """
        pair_predicate_question_samples = [{
            "id": id + 1, 
            "q_predicates": ", ".join(predicates)
        } for id, predicates in enumerate(list_predicates_question)]
        pair_predicate_question_obj = Prompt(
            template=pair_predicate_question_template,
            input_variables=["id", "question"]
        )
        pair_predicate_question_obj.create_fewshot_template(
            pair_predicate_question_samples,
            prefix=""
        )
        pair_predicate_question_prompt = pair_predicate_question_obj.get_prompt({})

        # INSTRUCTION PROMPT
        INSTRUCTION_PROMPT = LOGIC_PROGRAM_EXTRACTION_PROMPTING_DEFINITION()

        # FINAL PROMPT
        final_prompt_obj = Prompt(
            template=llama2_chat_prompt_template,
            input_variables=['instruct_prompt', 'user_question']
        )
        final_prompt_obj.create_prompt_template()

        # final_prompt = final_prompt_obj.get_prompt({
        #     'instruct_prompt': INSTRUCTION_PROMPT, 
        #     'user_question': f"{pair_predicate_premise_prompt}\n{pair_predicate_question_prompt}",
        # })

        final_prompt = final_prompt_obj.get_prompt({
            'instruct_prompt': INSTRUCTION_PROMPT, 
            'user_question': pair_predicate_premise_prompt,
        })
        return final_prompt


def check_multiple_question(config, device):
    # Load dataset path
    print("Load dataset")
    reasoning_dataset = XAIDataset(config['data']['train'], num_samples='all')

    for i in range(len(reasoning_dataset)):
        ques = reasoning_dataset[i]['conclusion']
        answer = reasoning_dataset[i]['answer']
        if len(answer.split(',')) >= 2:
            ic(ques, answer)


def parse_logic_program(logic_programs: list):
    '''
        Yield: predicate, natural languages
    '''
    for logic_program in logic_programs:
        pairs = logic_program.split(':::')
        predicate = pairs[0].strip()
        nl = pairs[1].strip() if len(pairs) == 2 else None
        yield predicate, nl


def extract_lp(model, info, config):
    print("Extract Logic Program")
    save_dict = {}
    chat_agent_extract_lp = ChatAgentExtractLogicProgram(model, config)  
    
    # Logic Programs
    premises = info['premises-nl'] # List
    questions = info['questions']
    llm_fol = info['LLM-FOL']
    ques_fols = info['question-FOL']


    # if type(questions) != list:
    #     questions = [questions]
    # if type(ques_fol) != list:
    #     ques_fol = [ques_fol]
    new_ques_fols = []
    new_questions = []
    for question, ques_fol in zip(questions, ques_fols):
        ic(question)    
        new_ques_fol = []
        new_ques = []
        if '\nA' in question or '\nB' in question or '\nC' in question or '\nD' in question:
            mlc_fol = ques_fol.split('\n')[1:]
            mlc_nl = question.split('\n')[1:]
            new_ques_fol.extend(mlc_fol)
            new_ques.extend(mlc_nl)
        new_ques_fols.extend(new_ques_fol)
        new_questions.extend(new_ques)
    
    # llm_fol.extend(ques_fols)
    # premises.extend(questions)
    ic(new_ques_fols)
    ic(new_questions)
    llm_fol.extend(new_ques_fols)
    premises.extend(new_questions)

    if "" in llm_fol:
        llm_fol.remove("")
    if "" in premises:
        premises.remove("")

    ic(premises)
    ic(len(llm_fol), len(premises))

    list_predicates_premise = [extract_predicate_from_fol(fol) for fol in llm_fol]
    list_predicates_question = [extract_predicate_from_fol(fol) for fol in ques_fol]

    new_list_predicates_premise = []
    new_list_premises = []
    exist_predicate_name = []
    for premise, list_predicates in zip(premises, list_predicates_premise):
        list_names = [predicate.split("(")[0] for predicate in list_predicates if predicate.split("(")[0] not in exist_predicate_name]
        list_predicates = [predicate for predicate in list_predicates if predicate.split("(")[0] not in exist_predicate_name]
        exist_predicate_name.extend(list_names)
        if len(list_names) == 0:
            continue
        new_list_predicates_premise.append(list_predicates)
        new_list_premises.append(premise)
    
    # Input question
    # final_prompt = chat_agent_extract_lp.make_prompt(
    final_prompt = chat_agent_extract_lp.make_prompt(
        premises=new_list_premises,
        list_predicates_premise=new_list_predicates_premise,
        list_predicates_question=list_predicates_question,
    )
    ic(final_prompt)
    for num_regenerate in tqdm(range(2), desc=("Extracting logic program")):
        extract_lp_results = chat_agent_extract_lp.inference(
            prompt=final_prompt,
            input_values={},
        )
        output = extract_lp_results['text'].split("[/INST]")[-1]
        ic(output)
        map_info = get_lp_info_v2(output)
        map_info = [f'{k} ::: {v}' for k, v in map_info.items() if k.split('(')[0] in exist_predicate_name]
        if not len(map_info) == 0:
            break
    return map_info

if __name__=="__main__":
    begin = time.time()
    args = get_args()
    config = load_yml(args.config)
    config['file_path'] = args.file_path
    extract_lp(config, args.device)
    # check_multiple_question(config, args.device)
    end = time.time()
    execute_time = end - begin
    ic(execute_time)
