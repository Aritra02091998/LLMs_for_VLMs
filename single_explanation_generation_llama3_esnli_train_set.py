'''
This script is used to generate possible explanation of each labels (Entailment, Conclusion, Neutral) 
of the ESNLI-VE dataset analysing the generated information like image captions, Objects in the image,
Hypotheis, Conclusion, Retrieved Facts from OMCS Knowledge Corpus using ColBERT Retriever and using LLM's (LLAMA3-8B here)
'''

import json
import csv
import os
import faiss
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
train_annt = pd.read_csv('../esnli/csv/esnlive_train.csv')
val_annt = pd.read_csv('../esnli/csv/esnlive_dev.csv')
test_annt = pd.read_csv('../esnli/csv/esnlive_test.csv')

# Extracting Data of Training Set

questionList_train = train_annt['hypothesis'].tolist()
knowledgeList_train = train_annt['explanation'].tolist()
answerList_train = train_annt['gold_label'].tolist()
imgList_train = train_annt['Flickr30kID'].tolist()

print('\nLength of the Set Given is:', len(questionList_train))


# Extracting Captions

captionsTrain = []

with open('../esnliImageCaptions/blip_2_captions/blip_image_captions_full_train_set.txt', 'r') as file:
    
    for lines in file:
        captionsTrain.append(lines.strip().split('_')[1])


# Extracting Objects

objectsListTrain = []

with open('../esnliObjectsList/esnlive_objects_train_set_full.txt', 'r') as file:
    
    for lines in file:
        objectsListTrain.append( (lines.strip().split(','))[:-1] )


# Extracting Relevant Facts Associated of Training Set

retrievedFactsFromTrainSet = []

with open('./esnliStoredFacts/esnliTrainStoredFacts.txt', 'r') as file:
    for lines in file:
        retrievedFactsFromTrainSet.append(lines.strip().split(',')[:-1])



# Creation of LLAMA3 Prompt using all the informations.
print('\nGenerating Explanation......\n')
with open('./llama3_single_explanation_generated.txt', 'w') as file:

	for idx in tqdm(range(0, len(questionList_train))):

		prompt= f'''
		I have an image, and I need to determine whether a given hypothesis is an entailment, contradiction, or neutral based on the information provided. Below are the details:

		Image Description:

		Caption: {captionsTrain[idx]}.
		Objects Present: {objectsListTrain[idx]}
		Hypothesis:

		{questionList_train[idx]}.
		Relevant Knowledge Facts:

		{retrievedFactsFromTrainSet[idx][0]}.
		{retrievedFactsFromTrainSet[idx][1]}.
		{retrievedFactsFromTrainSet[idx][2]}.
		{retrievedFactsFromTrainSet[idx][3]}.
		{retrievedFactsFromTrainSet[idx][4]}.

		Conclusion: The hypothesis is {answerList_train[idx]}.

		You will be rewarded $1000 to provide a single line on point explanation within 15-20 words. 
		Dont mention about any of the term "'"Explanation", "Hypothesis", "knowledge fact", "relevant fact", 'image description'
		'''

		# Write the generated explanation into text file

		file.write(f'{idx}_{generate_output(prompt)}\n')

print('\n\n****Exiting Script****\n')
