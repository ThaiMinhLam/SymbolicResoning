import os, sys, json, time, argparse
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from icecream import ic
from pprint import pprint

from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder,
)

sys.path.append("/data/npl/ViInfographicCaps/Contest/final_contest/XAI")
from utils import load_yml, load_llm, load_json, save_json
from src.dataloader import XAIDataset, load_dataloader 
from src.chat_agent import ChatAgent, Prompt
from src.module.reasoning.template_temp2 import PARSE_QUESTION, UNDERSTAND_BACKGROUND_PROMPT_EN

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/reasoning_config.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device to use")
    parser.add_argument("--file_path", type=str, help="Path to data file")
    return parser.parse_args()

class ReasoningDataset:
    def __init__(self, config: dict, split: str = "test", num_samples: int | str = "all"):
        self.dataset = XAIDataset(config, split)
        if num_samples != "all":
            num_samples = int(num_samples)
            if len(self.dataset) >= num_samples:
                self.dataset = self.dataset[:num_samples]

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

class ChatAgentReasoning:
    def __init__(self, config: dict, device: str = "cuda:0"):
        self.llm = load_llm(config, device)
        self.prompt_template = Prompt()

    def make_prompt(self, sample: dict) -> tuple[str, list[dict]]:
        messages = [
            HumanMessagePromptTemplate.from_template(PARSE_QUESTION),
            AIMessagePromptTemplate.from_template("Sure! Here is the logic representation of your question:\n{logic_program}"),
            HumanMessagePromptTemplate.from_template(UNDERSTAND_BACKGROUND_PROMPT_EN)
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        final_prompt = prompt.format_messages(
            question=sample['question'],
            background=sample['context'],
            logic_program=sample['output']['logic_question']
        )
        return final_prompt, sample['output']['logic_question']

    def __call__(self, prompt: list[dict]) -> str:
        return self.llm.invoke(prompt).content

def parse_logic_program(logic_programs: list[str]) -> list[tuple[str, str]]:
    result = []
    for logic_program in logic_programs:
        parts = logic_program.split(":::")
        predicate = parts[0].strip()
        nl = parts[1].strip() if len(parts) == 2 else ""
        result.append((predicate, nl))
    return result

def reasoning(config: dict, device: str = "cuda:0"):
    agent = ChatAgentReasoning(config, device=device)
    reasoning_dataset = ReasoningDataset(config)

    outputs = {}
    test_index = 1  # change this to run other examples

    for i in range(len(reasoning_dataset)):
        if i != test_index:
            continue

        sample = reasoning_dataset[i]
        prompt, logic_question = agent.make_prompt(sample)

        ic("Final Prompt", prompt)
        ic("Number of tokens", len(" ".join([p.content for p in prompt]).split()))

        response = agent(prompt)
        ic("Response", response)

        outputs[i] = {
            "original_question": sample['question'],
            "generated_logic_program": response,
            "gold_logic_program": parse_logic_program(logic_question),
        }

        save_json(outputs, os.path.join(config['output_dir'], "reasoning_output.json"))

if __name__ == "__main__":
    start = time.time()

    args = get_args()
    config = load_yml(args.config)
    config['file_path'] = args.file_path

    reasoning(config, args.device)

    ic(f"Execution Time: {time.time() - start:.2f} seconds")
