import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import load_json

class XAIDataset(Dataset):
    def __init__(self, annotation_path, num_samples='all'):
        self.annotation = load_json(annotation_path)
        self.data = self.sampling(num_samples=num_samples)
        self.clean_fol_premises()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def clean_fol_premises(self):
        list_fol_premises = [
            fol_premise.replace("FORALL", "ForAll") 
            if 
                'FORALL' in fol_premise 
            else fol_premise for fol_premise in self.data]

        list_fol_premises = [
            fol_premise.replace("ForAll", "∀x ") 
            if 
                'ForAll' in fol_premise 
            else fol_premise for fol_premise in list_fol_premises]

        list_fol_premises = [
            fol_premise.replace("Exists", "∃x ") 
            if 
                'ForAll' in fol_premise 
            else fol_premise for fol_premise in list_fol_premises]
        
        self.data = list_fol_premises

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
            premises = ' '.join(item_value['premises-NL'])
            fol_premises = '.'.join(item_value['premises-FOL'])
            questions = item_value['questions']
            answers = item_value['answers']
            reasonings = item_value['explanation']

            # Create samples
            for q_id, (question, answer, reasoning) in enumerate(zip(questions, answers, reasonings)):
                sample_item = {
                    'id': id,
                    'q_id': q_id,
                    'premises': premises,
                    'fol_premises': fol_premises,
                    'conclusion': question,
                    'reasoning': reasoning,
                    'answer': answer,
                }
                samples.append(sample_item)
                num_records += 1
            

                if num_samples != "all" and num_records >= num_samples:
                    return samples
        return samples
    
    

def collate_fn(batch):
    list_id = [item['id'] for item in batch]
    list_q_id = [item['q_id'] for item in batch]
    list_premises = [item['premises'] for item in batch]
    list_fol_premises = [item['fol_premises'] for item in batch]
    list_conclusion = [item['conclusion'] for item in batch]
    list_reasoning = [item['reasoning'] for item in batch]
    list_answer = [item['answer'] for item in batch]

    return list_id, list_q_id, list_premises, list_fol_premises, list_conclusion, list_reasoning, list_answer


def load_dataloader(dataset, batch_size=1, shuffle=True):
    dataloader = DataLoader(
        dataset, 
        shuffle=shuffle, 
        collate_fn=collate_fn, 
        batch_size=batch_size,
    )
    return dataloader