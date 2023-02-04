import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re
import json


class Decoder:
    def __init__(self, has_encoder=False, dic="decode_dict.json"):
        self.type = 'encoder' if has_encoder else 'simple'
        if self.type == 'encoder':
            self.decode_dict = json.load(open(dic, 'r'))
            self.decode_pattern = re.compile(r"\{([a-z \{\}\?]{1,2})\}")
    def decode(self, string):
        if self.type == 'simple':
            return string
        
        return re.sub(self.decode_pattern, lambda m: self.decode_dict[m.group(0)], string).capitalize()
    
print("Initializing model...\n")

# If you have save the model locally use these commands to import the model
tokenizer = GPT2Tokenizer.from_pretrained("/opt/NLP_Models/gpt2-model/", local_files_only=True)
model = GPT2LMHeadModel.from_pretrained("/opt/NLP_Models/gpt2-model/", local_files_only=True)

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')

charset = [' ', 'a', 'i', 'n', 'm', '{', '}', 'd', 'l', 's', 'r', 'h', 'o', 'u', 'e', 'y', 'b', 'x', 't', 'g', 'j', 'z', 'v', 'k', 'c', 'p', 'f', '?', 'w', 'q']
init_len = 5
decoder = Decoder(has_encoder=os.path.exists("decode_dict.json"))

text = input("Input the output of the model to reveal the message: ")

in_seq = torch.tensor(tokenizer.encode(text))
context = in_seq[:init_len]
msg = ""
k=len(charset)
with torch.no_grad():
    for i in range(init_len, in_seq.shape[0]):
        output = model(context)
        msg += charset[torch.where(torch.topk(output[0][-1, :], k).indices == in_seq[i])[0][0].item()]
        context = torch.cat((context, in_seq[i].unsqueeze(0)))

print("\nGenerated output: ")
print(decoder.decode(msg))