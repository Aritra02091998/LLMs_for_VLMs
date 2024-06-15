'''
This script is used to generate possible explanation of each labels (Entailment, Conclusion, Neutral) 
of the ESNLI-VE dataset analysing the generated information like image captions, Objects in the image,
Hypotheis, Conclusion, Retrieved Facts from OMCS Knowledge Corpus using ColBERT Retriever and using LLM's (LLAMA3-8B here)
'''

import json
import csv
import os
import torch
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm
import transformers

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initializing LLAMA_3

print('\nInitializing LLAMA3')
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    token = 'YOUR API KEY',
    pad_token_id = 128009
)

# Using device_map: "auto" will use GPU space from all the available GPU's.
# To use specific gpus, use cuda:0 or cuda:1

def generate_output(prompt):
    
    messages = [
        {"role": "system", "content": "Through reasoning, given a context and a conclusion, you can find the most plausible explanatuon"},
        {"role": "user", "content": prompt},
    ]
    
    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )
    
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = pipeline(
        prompt,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    return outputs[0]["generated_text"][len(prompt):]


# Extraction of ESNLI-VE
print('\nExtracting ESNLI-VE')
val_annt = pd.read_csv('../esnli/csv/esnlive_dev.csv')

# Extracting Data of Validation Set

questionList_val = val_annt['hypothesis'].tolist()
knowledgeList_val = val_annt['explanation'].tolist()
answerList_val = val_annt['gold_label'].tolist()
imgList_val = val_annt['Flickr30kID'].tolist()

print('\nLength of the Set Given is:', len(questionList_val))


# Extracting Captions

captionsVal = []

with open('../esnliImageCaptions/blip_2_captions/blip_image_captions_full_val_set.txt', 'r') as file:
    
    for lines in file:
        captionsVal.append(lines.strip().split('_')[1])


# Extracting Objects

objectsListVal = []

with open('../esnliObjectsList/esnlive_objects_val_set_full.txt', 'r') as file:
    
    for lines in file:
        objectsListVal.append( (lines.strip().split(','))[:-1] )


# Extracting Relevant Facts Associated of Training Set

retrievedFactsFromValSet = []

with open('../esnliStoredFacts/esnliValStoredFacts.txt', 'r') as file:
    for lines in file:
        retrievedFactsFromValSet.append(lines.strip().split(',')[:-1])


# Creation of LLAMA3 Prompt using all the informations.
print('\nGenerating Explanation......\n')
with open('./llama3_single_explanation_generated_val_set.txt', 'w') as file:

	for idx in tqdm(range(0, len(questionList_val))):

		prompt= f'''
		I have an image, and I need to determine whether a given hypothesis is an entailment, contradiction, or neutral based on the information provided. Below are the details:

		Image Description:

		Caption: {captionsVal[idx]}.
		Objects Present: {objectsListVal[idx]}
		Hypothesis:

		{questionList_val[idx]}.
		Relevant Knowledge Facts:

		{retrievedFactsFromValSet[idx][0]}.
		{retrievedFactsFromValSet[idx][1]}.
		{retrievedFactsFromValSet[idx][2]}.
		{retrievedFactsFromValSet[idx][3]}.
		{retrievedFactsFromValSet[idx][4]}.

		Conclusion: The hypothesis is {answerList_val[idx]}.

		You will be rewarded $1000 to provide a single line on point explanation within 15-20 words. 
		Dont mention about any of the term "'"Explanation", "Hypothesis", "knowledge fact", "relevant fact", 'image description'
		'''

		# Write the generated explanation into text file

		file.write(f'{idx}_{generate_output(prompt)}\n')

print('\n\n****Exiting Script****\n')
