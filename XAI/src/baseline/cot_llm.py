import json
import torch
import os
import argparse
import numpy as np
import pandas as pd
from icecream import ic

from dotenv import load_dotenv, dotenv_values 
from tqdm import tqdm

# Trackkkk
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
from utils import load_yml, load_llm
from src.dataloader import XAIDataset, load_dataloader 
from src.chat_agent import ChatAgent, Prompt

load_dotenv()

def get_args():
    parser = argparse.ArgumentParser(description="Load model config and run something")
    
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--device', type=int, required=True, default=1, help='Path to YAML config file')
    
    return parser.parse_args()


class ChatAgentCoT(ChatAgent):
    def __init__(self, config, device):
        super().__init__(config, device)

    def make_prompt_v2(self, fewshot_examples):
        '''
            Make a CoT Fewshot templates:
            Parameters
            ----------

            fewshot_examples = [list]
                List of example with {
                    'id': ...,
                    'q_id': ...,
                    'premises': ...,
                    'fol_premises': ...,
                    'conclusion': ...,
                    'reasoning': ...,
                    'judgement': ...,
                }
        '''


        example_template = """
            ### Input:
            "Premises": "{premises}"
            "Hypothesis": "{conclusion}"

            ### Response:
            "Thoughts": "Let us think step by step. {reasoning}"
            "Recall the Hypothesis": "{conclusion}"
            "Judgement": "Now we know that the Hypothesis is {judgement}"
            ---
        """

        llama2_chat_prompt_template = """
            <s>[INST] <<SYS>>
            ### Instruction:
            {instruct_prompt}
            ---

            {fewshot_examples}
            ### EXAMPLES

            <</SYS>>

            {user_question} [/INST]
        """

        # Prompt
        instruct_prompt = """
            Suppose you are one of the greatest AI scientists, logicians and mathematicians. Let us think step by step. 
            Read and analyze the "Premises" first, then using First-Order Logic (FOL) to judge whether the "Hypothesis":
                - If the question is binary-type question: The "Judgement" is Yes, No or Uncertain.
                - If the question is multiple-choice question: The "Judgement" is A, B, C or D.
                - If the question requiring a specific value, (e.g., What is the minimum GPA required?): The "Judgement" is a number.
                - If the question requiring multiples values, (e.g., How many students has more than 8 scores?): The "Judgement" is unlimited list of number
            Please make sure your reasoning is directly deduced from the "Premises" other than introducing unsourced common knowledge and unsourced information by common sense reasoning.
        """

        question_template = """
            Answer the question based only the "Premises" and "Hypothesis".
            Read and analyze the "Premises" but not using First-Order Logic (FOL) to
            - Give "Thoughts" clearly step-by-step with approriate premise's id.
            - Judge whether the "Hypothesis":
                - If the question is binary-type question: The "Judgement" is Yes, No or Uncertain.
                - If the question is multiple-choice question: The "Judgement" is A, B, C or D.
                - If the question requiring a specific value, (e.g., What is the minimum GPA required?): The "Judgement" is a number.
                - If the question requiring multiples values, (e.g., How many students has more than 8 scores?): The "Judgement" is unlimited list of number
            Please make sure your reasoning is directly deduced from the "Premises" other than introducing unsourced common knowledge and unsourced information by common sense reasoning.

            ### Input:
            "Premises": "{q_premises}"
            "Hypothesis": "{q_question}"

            $$$$
            ### Answer Response:
            "Thoughts": "Let us think step by step: "
            "Judgement": "Now we know that the correct answer is: "
            
        """

        # Prompting
        example_prompt = PromptTemplate(
            input_variables=["premises", "conclusion", "reasoning", "judgement"],
            template=example_template,
        )

        fewshot_prompt = FewShotPromptTemplate(
            examples=fewshot_examples,
            example_prompt=example_prompt,
            suffix="",
            input_variables=[],
        )

        # Format for final prompt
        llama2_chat_prompt = PromptTemplate(
            input_variables=["instruct_prompt", "fewshot_examples", "user_question"],
            template=llama2_chat_prompt_template,
        )

        llama2_chat_prompt_format = llama2_chat_prompt.format(
            instruct_prompt=instruct_prompt,
            fewshot_examples=fewshot_prompt.format(),
            user_question=question_template,
        )
        return llama2_chat_prompt_format

    def make_prompt(self, fewshot_examples):
        '''
            Make a CoT Fewshot templates:
            Parameters
            ----------

            fewshot_examples = [list]
                List of example with {
                    'id': ...,
                    'q_id': ...,
                    'premises': ...,
                    'fol_premises': ...,
                    'conclusion': ...,
                    'reasoning': ...,
                    'judgement': ...,
                }
        '''


        example_template = """
            ### Input:
            "Premises": "{premises}"
            "Hypothesis": "{conclusion}"

            ### Response:
            "Thoughts": "Let us think step by step. {reasoning}"
            "Recall the Hypothesis": "{conclusion}"
            "Judgement": "Now we know that the Hypothesis is {judgement}"
            ---
        """

        llama2_chat_prompt_template = """
            <s>[INST] <<SYS>>
            ### Instruction:
            {instruct_prompt}
            ---

            ### EXAMPLES
            {fewshot_examples}

            <</SYS>>

            {user_question} [/INST]
        """

        # Prompt
        instruct_prompt = """
            Suppose you are one of the greatest AI scientists, logicians and mathematicians. Let us think step by step. 
            Read and analyze the "Premises" first, then using First-Order Logic (FOL) to judge whether the "Hypothesis":
                - If the question is binary-type question: The "Judgement" is Yes, No or Uncertain.
                - If the question is multiple-choice question: The "Judgement" is A, B, C or D.
                - If the question requiring a specific value, (e.g., What is the minimum GPA required?): The "Judgement" is a number.
                - If the question requiring multiples values, (e.g., How many students has more than 8 scores?): The "Judgement" is unlimited list of number
            Please make sure your reasoning is directly deduced from the "Premises" other than introducing unsourced common knowledge and unsourced information by common sense reasoning.
        """

        question_template = """
            Answer the question based only the "Premises" and "Hypothesis".
            Read and analyze the "Premises" but not using First-Order Logic (FOL) to
            - Give "Thoughts" clearly step-by-step with approriate premise's id.
            - Judge whether the "Hypothesis":
                - If the question is binary-type question: The "Judgement" is Yes, No or Uncertain.
                - If the question is multiple-choice question: The "Judgement" is A, B, C or D.
                - If the question requiring a specific value, (e.g., What is the minimum GPA required?): The "Judgement" is a number.
                - If the question requiring multiples values, (e.g., How many students has more than 8 scores?): The "Judgement" is unlimited list of number
            Please make sure your reasoning is directly deduced from the "Premises" other than introducing unsourced common knowledge and unsourced information by common sense reasoning.

            ### Input:
            "Premises": "{q_premises}"
            "Hypothesis": "{q_question}"

            $$$$
            ### Answer Response:
            "Thoughts": "Let us think step by step: "
            "Judgement": "Now we know that the correct answer is: "
            
        """

        # Prompting
        fewshot_prompt_obj = Prompt(
            template=example_template,
            input_variables=["premises", "conclusion", "reasoning", "judgement"]
        )
        fewshot_prompt_obj.create_fewshot_template(fewshot_examples)
        fewshot_prompt = fewshot_prompt_obj.get_prompt({})


        # Format for final prompt
        llama2_chat_prompt_obj = Prompt(
            template=llama2_chat_prompt_template,
            input_variables=["instruct_prompt", "fewshot_examples", "user_question"],
        )
        llama2_chat_prompt_obj.create_prompt_template()
        llama2_chat_prompt = llama2_chat_prompt_obj.get_prompt({
            'instruct_prompt': instruct_prompt,
            'fewshot_examples': fewshot_prompt,
            'user_question': question_template,
        })

        return llama2_chat_prompt
    
def main(config):
    print("Load dataset")
    # Load dataset
    train_dataset = XAIDataset(config['data']['train'], num_samples='all')
    val_dataset = XAIDataset(config['data']['val'], num_samples=100)
    test_dataset = XAIDataset(config['data']['test'], num_samples=100)

    train_dataloader = load_dataloader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
    )
    val_dataloader = load_dataloader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=True,
    )
    test_dataloader = load_dataloader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
    ) 

    print("Load Chat Agent")
    # Load ChatAgent
    model = load_llm(
        model_id=config['model_id'],
        config=config['model_config'],
        model_type=config['model_type'],
        device=config.device,
    )
    chat_agent = ChatAgentCoT(model, config)
    fewshot_examples = train_dataset.data[:2]
    cot_prompt = chat_agent.make_prompt(fewshot_examples)

    print("Chain example")
    # Inference one sample:
    for list_id, list_q_id, list_premises, list_fol_premises, list_conclusion, list_reasoning, list_judgement in train_dataloader:
        input_values = {
            'q_question': list_conclusion[0], # Take the first element in batch
            'q_premises': list_premises[0],  # Take the first element in batch
        }
        
        results = chat_agent.inference(
            prompt=cot_prompt,
            input_values=input_values,
        )

        print(results['text'])
        break

if __name__=="__main__":
    args = get_args()
    # ic(args.device.split(':')[-1])
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[-1]
    config = load_yml(args.config)
    main(config)
