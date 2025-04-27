'''
    This code split the train set of XAI trainingset to different files based-on FOL-premises types:
    
    ----
    FOL-types
        - LP    : 
            + Having a "Facts" that  formulated as P(a1, · · · , an) - Examples: 
                > Age(Peter, 31)            : Peter age is 31
                > MadeOfIron(Nails, True)   : Nails are made of iron
                
        - FOL   : Having letter ∀, ∃
        - 
        -

'''


import sys
sys.path.append("/data/npl/ViInfographicCaps/Contest/final_contest/XAI")

from src.dataloader import XAIDataset
from utils import load_yml

config = load_yml("/data/npl/ViInfographicCaps/Contest/final_contest/XAI/config/config_llama2.yml")
train_dataset = XAIDataset(config['data']['train'], num_samples='all')