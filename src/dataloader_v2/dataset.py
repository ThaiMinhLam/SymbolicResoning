import torch
import sys
sys.path.append("/data/npl/ViInfographicCaps/Contest/final_contest/XAI")
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils.utils import load_json

class XAIDataset(Dataset):
    def __init__(self, annotation_path, num_samples='all'):
        self.annotation = load_json(annotation_path)
        self.data = self.sampling(num_samples=num_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

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
            premises = item_value['premises-NL']
            fol_premises = item_value['premises-FOL']
            questions = item_value['questions']
            answers = item_value['answers']
            reasonings = item_value['explanation']


            sample_item = {

                'premises-nl': premises,
                'fol_premises': fol_premises,
                'questions': questions,
                'reasonings': reasonings,
                'answer': answers,
            }


            samples.append(sample_item)
            num_records += 1
        

            if num_samples != "all" and num_records >= num_samples:
                return samples
    
        return samples
    
    

def collate_fn(batch):
    list_premises = []
    list_fol_premises = []
    list_conclusion = []
    list_reasoning = []
    list_answer = []

    for item in batch:
        list_premises.append(item['premises-nl'])
        list_fol_premises.append(item['fol_premises'])
        list_conclusion.append(item['questions'])
        list_reasoning.append(item['reasonings'])
        list_answer.append(item['answer'])

    return {
        'premises-nl': list_premises,
        'fol_premises': list_fol_premises,
        'questions': list_conclusion,
        'reasonings': list_reasoning,
        'answers': list_answer,
    }


def load_dataloader(dataset, batch_size=1, shuffle=False):
    dataloader = DataLoader(
        dataset, 
        shuffle=shuffle, 
        collate_fn=collate_fn, 
        batch_size=batch_size,
    )
    return dataloader