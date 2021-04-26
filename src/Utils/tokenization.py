
from transformers import BertTokenizerFast

def bert_tokenize(args):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased',do_lower_case=args.do_lower_case)
    return tokenizer