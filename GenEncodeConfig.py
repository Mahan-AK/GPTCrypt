import sys
import pandas as pd
import json
from collections import Counter


supported_chars = [' ', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '}']

assert len(sys.argv)>1, "Path of the text file must be given as an argument!"
with open(sys.argv[1], 'r') as f:
    text = f.read()

words = [word.strip(",.?\'\":;!").lower() for line in text.split('\n') for word in line.split(' ')]

word_counts = pd.Series(Counter(words))
char_counts = pd.Series(Counter(text))
char_counts.sort_values(ascending=False, inplace=True)

encodables = pd.concat([word_counts, char_counts])
encodables = encodables[list((set(encodables.index) - set(supported_chars)) - set([ch.upper() for ch in supported_chars]))].sort_values(ascending=False)

freq1 = encodables[[len(w)>3 or len(w)==1 for w in encodables.index]][:len(supported_chars)].index
encode_dict = {k:'{'+v+'}' for k, v in zip(freq1, supported_chars)}

encodables = encodables[list((set(encodables.index) - set(freq1)))].sort_values(ascending=False)
one_chars = encodables[[len(w)==1 for w in encodables.index]]
word_enc = encodables[[len(w)>4 for w in encodables.index]][:(len(supported_chars)**2-len(one_chars))]

perm_scores = pd.Series({ch1+ch2:char_counts[ch1]*char_counts[ch2] for ch1 in supported_chars for ch2 in supported_chars})
ranked_perm = list(perm_scores.sort_values(ascending=False).index)

freq2 = list(pd.concat([one_chars, word_enc]).sort_values(ascending=False).index)
encode_dict2 = {k:'{'+v+'}' for k, v in zip(freq2, ranked_perm)}

encode_dict.update(encode_dict2)
decode_dict = {v:k for k, v in encode_dict.items()}

json.dump(encode_dict, open("encode_dict.json", 'w'))
json.dump(decode_dict, open("decode_dict.json", 'w'))