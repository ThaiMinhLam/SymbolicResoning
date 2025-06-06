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
from utils import load_yml, load_llm, load_json, save_json, get_main_predicate
from src.dataloader import XAIDataset, load_dataloader 
from src.chat_agent import ChatAgent, Prompt
from src.module.reasoning import EXTRACT_MAIN_PREDICATE, HEAD_INSTRUCTION
load_dotenv()

'''
    LINA MODEL Explain:
    -  Apply hypothetical-deductive method (LLM + FOL) on "question"
    
    HAVING 2 MODULES:
    ----
    -  Information Extraction: (1)

    ----
    -  LLM-driven Symbolic Reasoning: (2)
       
'''

def get_args():
    parser = argparse.ArgumentParser(description="Load model config and run something")
    
    parser.add_argument('--examples', type=str, required=True, help='Path to Examples Json File')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--device', type=int, required=True, default='cuda:0', help='Path to YAML config file')
    
    return parser.parse_args()


class ChatAgentParaphrase(ChatAgent):
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
        } for id, question in enumerate(list_questions)]
        statement_prompt_obj = Prompt(
            template=fs_item_template,
            input_variables=["id", "question"]
        )
        statement_prompt_obj.create_fewshot_template(
            fewshot_examples,
            prefix="Extract these following statements:\n")
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


def test_paraphrase(config, device):
    # Load Examples path
    save_dir = 'save'
    examples_file = load_json(config['example_json'])
    save_path = os.path.join(save_dir, 'paraphrase.json')
    # Load ChatAgent
    model = load_llm(
        model_id=config['model_id'],
        config=config['model_config'],
        model_type=config['model_type'],
        device=device,
    )

    print("Chain example")
    context_collection = [[nl for nl in item['premises-NL']] for item in examples_file]
    question_collection = [[ques for ques in item['questions']] for item in examples_file]

    chat_agent = ChatAgentParaphrase(model, config)
    context_input_values = {}
    paraphrase_content = {}
    dict_info = None
    for idx, (context, questions) in enumerate(zip(context_collection, question_collection)):
        # ic(context)
        paraphrase_prompt = chat_agent.make_prompt(context)
        for num_regenerate in tqdm(range(5), desc=("Generate response")):
            context_results = chat_agent.inference(
                prompt=paraphrase_prompt,
                input_values=context_input_values,
            )

            response = context_results['text']
            ic(response)
            dict_info = get_main_predicate(response)
            dict_info['statement'] = context
            dict_info['questions'] = questions
            ic(dict_info)
            if len(dict_info['list predicates']) != 0:
                break

        paraphrase_content[idx] = dict_info
        break
    save_json(paraphrase_content, save_path)

    # ic(context_results.split('$$$$')[-1])


if __name__=="__main__":
    begin = time.time()
    args = get_args()
    config = load_yml(args.config)
    config['example_json'] = args.examples
    test_paraphrase(config, args.device)
    end = time.time()
    execute_time = end - begin
    ic(execute_time)
