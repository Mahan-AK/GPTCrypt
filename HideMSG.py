import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re
import json
import random


class Encoder:
    def __init__(self, has_encoder=False, dic="encode_dict.json"):
        self.type = 'encoder' if has_encoder else 'simple'
        if self.type == 'encoder':
            self.encode_dict = json.load(open(dic, 'r'))
            self.encode_pattern = re.compile('|'.join(re.escape(k) for k in self.encode_dict))
    def encode(self, string):
        if self.type == 'simple':
            return re.sub(r"[^a-zA-Z \{\}\?]", '?', string).lower()

        return re.sub(self.encode_pattern, lambda m: self.encode_dict[m.group(0)], string.lower())
    
print("Initializing model...\n")

# If you have save the model locally use these commands to import the model
# tokenizer = GPT2Tokenizer.from_pretrained("/opt/NLP_Models/gpt2-model/", local_files_only=True)
# model = GPT2LMHeadModel.from_pretrained("/opt/NLP_Models/gpt2-model/", local_files_only=True)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

charset = [' ', 'a', 'i', 'n', 'm', '{', '}', 'd', 'l', 's', 'r', 'h', 'o', 'u', 'e', 'y', 'b', 'x', 't', 'g', 'j', 'z', 'v', 'k', 'c', 'p', 'f', '?', 'w', 'q']
inits = json.load(open("initializers.json", 'r'))
encoder = Encoder(has_encoder=os.path.exists("encode_dict.json"))

text = input("Input your desired text to hide: ")

while True:
    initializer = random.choice(inits)
    print("Chose a random initializer:", initializer)

    text = encoder.encode(text)
    context = torch.tensor(tokenizer.encode(initializer))

    print("\nGenerating output:")
    print(initializer, end='')
    k=len(charset)
    with torch.no_grad():
        for i in range(len(text)):
            output = model(context)
            token = torch.topk(output[0][-1, :], k).indices[charset.index(text[i])]
            context = torch.cat((context, token.unsqueeze(0)))
            print(tokenizer.decode(token), end='')

        sequence = tokenizer.decode(context)

    print()
    if tokenizer.encode(sequence) == context.tolist():
        break
    else:
        print("Generated output does not have a one to one tokenized/detokenized relationship. Retrying...")
    
