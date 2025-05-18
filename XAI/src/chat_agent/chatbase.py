import json
import torch
import argparse
import numpy as np
import pandas as pd

from dotenv import load_dotenv, dotenv_values 
from tqdm import tqdm
from icecream import ic

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
from utils import load_yml, load_llm
from src.dataloader import load_dataloader 


class ChatAgent():
    def __init__(self, model, config):
        self.config = config
        self.llm_model = model


    def batch_inference(self, prompt, questions):
        """
            - Variables depend on the input_variables of your llama_prompt
            llama_prompt = PromptTemplate(
                input_variables=["q_question", "q_premises"],
                template=prompt,
            )
            
            questions = [{
                "q_question": q_question,
                "q_premises": q_premises,
            }, ....]
        """
        llama_prompt = PromptTemplate(
            input_variables=list(questions[0].keys()),
            template=prompt,
        )

        qa_chains = LLMChain(
            llm=self.llm_model,
            prompt=llama_prompt,
        )

        qa_chains.batch_inference(questions, return_source_documents=False)


    def inference(self, prompt, input_values: dict):
        """
            - Variables depend on the input_variables of your llama_prompt
            llama_prompt = PromptTemplate(
                input_variables=["q_question", "q_premises"],
                template=prompt,
            )
            
            input_values = {
                "q_question": q_question,
                "q_premises": q_premises,
            }
        """
        llama_prompt = PromptTemplate(
            input_variables=list(input_values.keys()),
            template=prompt,
        )

        prompt_text = llama_prompt.format(**input_values)
        # ic(prompt_text)
        ic(len(prompt_text.split()))

        qa_chains = LLMChain(
            llm=self.llm_model,
            prompt=llama_prompt,
        )
        results = qa_chains.invoke(input_values, return_source_documents=False,)
        return results
    
    def inference_direct(self, prompt):
        prompt_template = PromptTemplate.from_template(prompt)
        qa_chains = LLMChain(
            llm=self.llm_model,
            prompt=prompt_template,
        )
        results = qa_chains.invoke({}, return_source_documents=False,)
        return results

    def make_prompt(self):
        NotImplemented


class Prompt():
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
        self.prompt_template = None


    def create_prompt_template(self):
        self.prompt_template = PromptTemplate(
            input_variables=self.input_variables,
            template=self.template,
        )
    
    def create_fewshot_template(self, examples: list, suffix="", prefix=""):
        self.create_prompt_template()
        self.prompt_template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=self.prompt_template,
            prefix=prefix,
            suffix=suffix,
            input_variables=[],
        )

    def get_prompt(self, input_keys_values: dict):
        return self.prompt_template.format(**input_keys_values)