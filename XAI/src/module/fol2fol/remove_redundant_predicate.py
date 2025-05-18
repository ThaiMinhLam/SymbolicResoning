import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
import re
from icecream import ic
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
from src.chat_agent import ChatAgent, Prompt
from src.module.fol2fol.template import EXTRACT_MAIN_PREDICATE, HEAD_INSTRUCTION, REDUCE_AND_MATCHING_PREDICATE_PROMPT
from utils import get_main_predicate, filter_similar_predicate, extract_predicate_from_fol, parse_map_predicate
load_dotenv()

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
            # input_variables=['instruct_prompt']
            input_variables=['instruct_prompt', 'user_question']
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


def find_main_predicate(model, config, premises):
    chat_agent = ChatAgentFindMainPredicate(model, config)
    dict_info = None
    # premises.extend(questions)
    paraphrase_prompt = chat_agent.make_prompt(premises)
    for num_regenerate in tqdm(range(2), desc=("Generate response")):
        context_results = chat_agent.inference(
            prompt=paraphrase_prompt,
            input_values={},
        )
        response = context_results['text']
        dict_info = get_main_predicate(response)
        if len(dict_info['list predicates']) != 0:
            combine_predicates = []
            for item in dict_info['list predicates']:
                combine_predicates.extend(item)
            filter_list = filter_similar_predicate(combine_predicates)
            main_predicates = ", ".join([f"{item}" for item in filter_list])
            return main_predicates
    return ""



def reducing(model, info, config):
    print("Start Mapping")
    chat_agent_reducing = ChatAgentReduce(model, config)    
    # Logic Programs
    logic_program_predicates = info['logic_program']
    lp_predicates_list = logic_program_predicates
    
    # Input question
    ## NOTE: Question chỗ này là từng question nhỏ chứ ko phải là list các question. Vô module chính chỉnh lại thành một list các question ban đầu
    final_prompt = chat_agent_reducing.make_prompt(
        lp_predicates_list=lp_predicates_list,
        main_predicates='',
    )

    # Parsing mapping
    for j in tqdm(range(3)):
        tqdm.write(f"Mapping values iter {j}")
        mapping_results = chat_agent_reducing.inference_direct(
            prompt=final_prompt,
        )
        ic(mapping_results)
        mapping_output = mapping_results['text'].split("[/INST]")[-1]
        mapping_dict = parse_map_predicate(mapping_output, cal_distance=True, threshold=0.51)
        if len(mapping_dict) != 0:
            break
    # save_dict['results'] = mapping_results
    # save_dict['maps'] = mapping_dict
    # save_dict['logic_program'] = logic_program_predicates
    # save_dict['fol'] = fols

    # Map to fol
    # new_fols = map_to_fol(save_dict)
    # return new_fols, mapping_dict
    return mapping_dict