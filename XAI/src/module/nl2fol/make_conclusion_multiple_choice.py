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
import re

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
from utils import load_yml, load_llm, load_json, save_json, extract_predicate_from_fol
from src.dataloader import XAIDataset, load_dataloader 
from src.chat_agent import ChatAgent, Prompt
from src.module.reasoning import MAKE_CONCLUSION_FROM_OPTION_QUESTION

'''
   Make conclusion base on question and options
       
'''

class ChatAgentMakeConclusion(ChatAgent):
    def __init__(self, model, config):
        super().__init__(model, config)

    def make_prompt(self, question, option):
        # PROMPT TEMPLATE
        llama2_chat_prompt_template = """
            <s>[INST] <<SYS>>
            ### Instruction:
            {instruct_prompt}

            <</SYS>>
            ### Question
            {user_question} [/INST]
        """

        # FINAL PROMPT
        INSTRUCTION_PROMPT = MAKE_CONCLUSION_FROM_OPTION_QUESTION()
        # INSTRUCTION_PROMPT = MAKE_CONCLUSION_FROM_OPTION_QUESTION_WITH_REG_DETAIL()
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
            'user_question': f'Question: {question} \nOption: {option}',
        })
        return final_prompt


def parse_factual_statement(response):
    text = response.split("<</SYS>>")[-1]
    pattern_factual_statement = r"Factual Statement: .*$"

    match_statement = re.findall(pattern_factual_statement, text, re.MULTILINE)
    statement = match_statement[0].replace("Factual Statement:", "").strip()
    return statement


def make_conclusion(model, question, config):
    print("Start Making Conclusion")
    chat_agent_make_conclusion = ChatAgentMakeConclusion(model, config)  

    def parse_options(text):
        parts = re.split(r'\n(?=[A-D]\.)', text.strip())
        parts = [re.sub(r'^[A-D]\.\s*', '', opt.strip()) for opt in parts]
        question = parts[0]
        options = parts[1:]
        return question, options

    options = []
    # Check multiple choice
    if len([True for option in ["\nA.", "\nB.", "\nC.", "\nD."] if option in question]) >= 2:
        question, options = parse_options(question)
    
    if len(options) == 0:
        print(question)
        raise Exception("There is not Multiple Choice Question")


    # Input question
    new_options = []
    option_labels = ["\nA.", "\nB.", "\nC.", "\nD."]
    for label, option in zip(option_labels, options):
        conclusion_prompt = chat_agent_make_conclusion.make_prompt(
            question=question.strip(),
            option=option.strip()
        )

        # ic(conclusion_prompt)
        make_conclusion_results = chat_agent_make_conclusion.inference(
            prompt=conclusion_prompt,
            input_values={},
        )
        response = make_conclusion_results['text']
        factual_statement = parse_factual_statement(response)
        new_options.append(f"{label} {factual_statement}")
    return " ".join(new_options)