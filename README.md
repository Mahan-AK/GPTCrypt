# GPTCrypt

`where there is choice, there is a way to store information`

This script uses the way transformer models generate output to hide a message in the output. The way it works is that the model internally generates a probability distribution of all the possible tokens and generally uses topk method to choose the best next token. We can use this choice to hide bits of data in the output.

## Requirements

This script currently uses the `GPT-2` model from OpenAI to function but other transformer models should be able to be used. In my experience this model was fairly fast and had sufficiently convincing outputs. `GPT-2` model can be found in the [huggingface page](https://huggingface.co/gpt2).

## Usage

Simply run the `HideMSG.py` script to hide a message:
```
$ python HideMSG.py
Initializing model...

Input your desired text to hide: where there is choice there is a way to store information
Chose a random initializer: Not only can it appease

Generating output:
Not only can it appease fans from different corners around the galaxy that wish there be more Star Fox movies from DC to be created as well to ensure StarWars II is as faithful to its original IP, fans can help with the design with various pieces which will allow them enjoy these unique features and take the characters and
```

You can reveal the coded message using `RevealMSG.py`:
```
$ python RevealMSG.py
Initializing model...

Input the output of the model to reveal the message: Not only can it appease fans from different corners around the galaxy that wish there be more Star Fox movies from DC to be created as well to ensure StarWars II is as faithful to its original IP, fans can help with the design with various pieces which will allow them enjoy these unique features and take the characters and

Generated output: 
where there is choice there is a way to store information
```

## Additional Notes

You can use script right away but it has a very limited character set to keep the model output cohesive:

* lowercase alphabetical characters: `a-z`
* characters to enable encoding other characters outside this character: `{}`
* to show unsupported characters: `?`

To use other unicode characters, an encoding dictionary will be necessary. This can be done simply by feeding a text corpus to `GenEncodeConfig.py` to generate the necessary files:

```
$ python GenEncodeConfig.py text_corpus.txt
```

The text corpus can be a chat log and can be in any preferred language with arbiterary character set. The generated dictionary will contain encodings for unsupported characters and long words found in the text corpus. This will enable using other unicode characters and also encode longer words like `congratulations` to a much shorter word like `{dq}`, improving the performance.