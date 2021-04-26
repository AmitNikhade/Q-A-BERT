
from transformers import BertForQuestionAnswering, BertConfig
import torch
import torch
from transformers import BertForQuestionAnswering



class QA:

    def __init__(self,args):
       
        self.model_path = args.model_path
        self.use_cache=args.use_cache
        self.c_path = args.config_file
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu' 
        self.mp = args.output_dir
        
    def load_model(self):
        config = BertConfig.from_pretrained(self.c_path)
        self.model = BertForQuestionAnswering.from_pretrained('bert-base-uncased', config=config)       
        self.model.to(self.device)
        self.model.eval()
        return self.model
    

   