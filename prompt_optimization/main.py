import os
import pandas as pd
import json
import string
import random
from pathlib import Path
from module.azure_openai import AzureOpenAI
from configparser import ConfigParser

import dspy
from dspy.primitives import Example

from teleprompt.bootstrap import BootstrapFewShot
from teleprompt.copro_optimizer import COPRO
from teleprompt.teleprompt import Teleprompter

from utils.docs2data import read_docs_to_dataframe
from utils.validation import substring_metric, lm_metric
from .signatures import init_signatures
from .scan_report import ScanReport

config_object = ConfigParser()
config_object.read("config.ini")
tud_api_key = config_object["TUD_API_KEY"]['tud_api_key']
dev_api_key = config_object["DEV_API_KEY"]['dev_api_key']


def generate_random_string(length):
    characters = string.digits
    return ''.join(random.choices(characters, k=length))



def optimize(type_of_documents, categories, class_of_categories, objective, metric, optimizer = 'bootstrap-few-shot'):


    # 1. Create training data from .txt documents, i.e. table with columns [filename, context, quote, categories, count_quote, count_categories, answer]

    # create df from txt reports
    folder_path = Path('data/reports/train_annotated/')
    df = read_docs_to_dataframe(folder_path)
    # df = df[df['quote'].apply(len) > 0]
    # Remove quotation marks
    df['context'] = df['context'].str.replace(r'["]', '', regex=True)
    # # df

    # df = pd.read_excel('prompt_optimization/train_data.xlsx')

    training_examples = json.loads(df[["context","answer"]].to_json(orient="records"))
    train_dspy = [dspy.Example(x).with_inputs('context') for x in training_examples]


    # 2. Configure LLM API

    # llm credentials
    lm_gpt = AzureOpenAI(
        tud_dev = "TUD",
        api_version = '2024-06-01', #'2024-06-01',#'2023-07-01-preview',
        model_name = "gpt-4o", 
        api_key = tud_api_key,
        model_type = "chat"
    )

    dspy.settings.configure(lm=lm_gpt)

    number_of_items_in_output = 5
    

    # 3. Create signatures, i.e. separate short parametrized model prompts that will be evaluated separately

    hint, ChunkerSignature, PredictRelevance, PredictCategory, Translator = init_signatures(type_of_documents, number_of_items_in_output, objective, 
        class_of_categories, categories)


    # 4. Create class ScanReport that bundles signatures together and implements a forward function that is used to predict based on given contexts. 

    scan = ScanReport(hint, ChunkerSignature, PredictRelevance, PredictCategory, Translator)

    if metric == 'simple':
        validation_metric = substring_metric
    elif metric == 'llm':
        validation_metric = lm_metric


    # 5. Training with optimizer 'bootstrap-few-shot' or a combination of 'bootstrap-few-shot' and COPRO . Result is a saved model

    if optimizer == 'bootstrap-few-shot':
        
        # Set up a basic teleprompter, which will compile our program.
        teleprompter = BootstrapFewShot(metric=validation_metric, max_rounds=4,max_bootstrapped_demos=5, max_errors=1)

        # Compile
        compiled_ScanReport = teleprompter.compile(scan, trainset=train_dspy)   
        file_name = generate_random_string(10)
        save_path = f'{os.getcwd()}/models/{file_name}.json'
        # save_path = f'models/{file_name}.json'
        compiled_ScanReport.save(save_path)
    
    elif optimizer == 'combined':

        # Set up COPRO
        # teleprompter = COPRO(metric=validation_metric, breadth=2, depth=2, init_temperature=0.9)
        eval_dict = {
            "display_progress": False,
            "display_table": 0
            }
        # Compile COPRO
        # compiled_ScanReport1 = teleprompter.compile(student = scan, trainset=train_dspy[0:6], eval_kwargs = eval_dict)

        print("COPRO round 1 done, optimize with BootstrapFewShot")

        # Set up BootstrapFewShot
        teleprompter = BootstrapFewShot(metric=validation_metric, max_rounds=4,max_bootstrapped_demos=6, max_errors=1)
        # Compile BootstrapFewShot
        # compiled_ScanReport2 = teleprompter.compile(compiled_ScanReport1, trainset=train_dspy)
        compiled_ScanReport2 = teleprompter.compile(scan, trainset=train_dspy)

        print("BootstrapFewShot done, optimize with COPRO round 2")

        # again COPRO
        # teleprompter = COPRO(metric=validation_metric, breadth=2, depth=2, init_temperature=0.9)
        teleprompter = COPRO(metric=validation_metric, breadth=3, depth=2, init_temperature=0.8)
        compiled_ScanReport = teleprompter.compile(student = compiled_ScanReport2, trainset=train_dspy[0:8], eval_kwargs = eval_dict)
        # compiled_ScanReport = teleprompter.compile(student = compiled_ScanReport2, trainset=train_dspy, eval_kwargs = eval_dict)

        file_name = generate_random_string(10)
        save_path = f'../models/{file_name}.json'
        compiled_ScanReport.save(save_path)

    
    print('Number of saved model: ', file_name)

    # # Set up the evaluator, which can be re-used in your code.
    # evaluator = Evaluate(devset=train_dspy[0:13], num_threads=1, display_progress=True, display_table=0)

    # # Launch evaluation.
    # res = evaluator(scan, metric=validation_metric, return_all_scores=True)

    return (save_path, file_name)

        

