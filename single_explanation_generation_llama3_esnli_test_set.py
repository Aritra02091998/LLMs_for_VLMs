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
test_annt = pd.read_csv('../esnli/csv/esnlive_test.csv')

# Extracting Data of Training Set

questionList_test = test_annt['hypothesis'].tolist()
knowledgeList_test = test_annt['explanation'].tolist()
answerList_test = test_annt['gold_label'].tolist()
imgList_test = test_annt['Flickr30kID'].tolist()

print('\nLength of the Set Given is:', len(questionList_test))


# Extracting Captions

captionsTest = []

with open('../esnliImageCaptions/blip_2_captions/blip_image_captions_full_test_set.txt', 'r') as file:
    
    for lines in file:
        captionsTest.append(lines.strip().split('_')[1])


# Extracting Objects

objectsListTest = []

with open('../esnliObjectsList/esnlive_objects_test_set_full.txt', 'r') as file:
    
    for lines in file:
        objectsListTest.append( (lines.strip().split(','))[:-1] )


# Extracting Relevant Facts Associated of Training Set

retrievedFactsFromTestSet = []

with open('../esnliStoredFacts/esnliTestStoredFacts.txt', 'r') as file:
    for lines in file:
        retrievedFactsFromTestSet.append(lines.strip().split(',')[:-1])



# Creation of LLAMA3 Prompt using all the informations.
print('\nGenerating Explanation......\n')
with open('./llama3_single_explanation_generated_test.txt', 'w') as file:

    for idx in tqdm(range(0, len(questionList_test))):

        prompt= f'''
        I have an image, and I need to determine whether a given hypothesis is an entailment, contradiction, or neutral based on the information provided. Below are the details:

        Image Description:

        Caption: {captionsTest[idx]}.
        Objects Present: {objectsListTest[idx]}
        Hypothesis:

        {questionList_test[idx]}.
        Relevant Knowledge Facts:

        {retrievedFactsFromTestSet[idx][0]}.
        {retrievedFactsFromTestSet[idx][1]}.
        {retrievedFactsFromTestSet[idx][2]}.
        {retrievedFactsFromTestSet[idx][3]}.
        {retrievedFactsFromTestSet[idx][4]}.

        Conclusion: The hypothesis is {answerList_test[idx]}.

        You will be rewarded $1000 to provide a single line on point explanation within 15-20 words. 
        Dont mention about any of the term "'"Explanation", "Hypothesis", "knowledge fact", "relevant fact", 'image description'
        '''

        # Write the generated explanation into text file

        file.write(f'{idx}_{generate_output(prompt)}\n')

print('\n\n****Exiting Script****\n')
