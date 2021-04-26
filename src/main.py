
import argparse
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizerFast

def finetuned_model():
    ft_model = BertForQuestionAnswering.from_pretrained('/home/lucifer/output/model')
    ft_tokenizer = BertTokenizerFast.from_pretrained('/home/lucifer/output/tokenizer')
    return ft_model, ft_tokenizer

def decode( args):
    context = str(open(args.text_file, 'rb'))
    question = str(input('Enter Question:'))
    m,t = finetuned_model()
    input_ids = t.encode(question,context)
    tokens = t.convert_ids_to_tokens(input_ids)
    sep_index = input_ids.index(t.sep_token_id)
    num_seg_a = sep_index + 1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)
    start_scores, end_scores = m(torch.tensor([input_ids]),token_type_ids=torch.tensor([segment_ids]), return_dict=False) 
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    answer = ' '.join(tokens[answer_start:answer_end+1])
    return answer

def main():
    desc = "Question Answering System Using BERT"
    
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument("--text_file", default=None, type=str,required=True,
                        help="path_to context file")
 
    args = parser.parse_args()
    print(decode(args))

if __name__ == "__main__":
    main()