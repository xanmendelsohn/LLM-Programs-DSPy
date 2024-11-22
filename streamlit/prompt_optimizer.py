import streamlit as st
import openai
from openai import AzureOpenAI
from io import StringIO
import httpx
from datetime import datetime
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname('/home/cdsw/CB2MUHX/project-weeks-promptu/prompt_optimization'))
sys.path.append(os.path.dirname('/home/cdsw/CB2MUHX/project-weeks-promptu/data/reports/'))

from prompt_optimization.main import optimize

# Configure Streamlit page 
st.set_page_config(page_title='Automatic Prompt Tuning with LLM', layout='wide', page_icon='assets/favicon.ico')

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('assets/style.css')

# Configure page header
col1, col2= st.columns([2,1], gap="small")
col2.image("assets/coba-logo.png", width=500)
col1.title('Automatic Prompt Optimizer')

# introduction
st.write("Welcome to your prompt optimizer! This tool helps you to create and automatically optimize prompts. You can customize your prompt by specifying the following input fields.")

st.write("---")

# Instruction specification
col1, col2 = st.columns([1,1])

col1.subheader("Document typ:", help="Please name the type of document you want to extract and classify information from, i.e. business report (singular)")
type_of_documents = col1.text_input("Insert document type:", value = 'business report')
st.session_state['type_of_documents'] = type_of_documents

col2.subheader("Category group:", help="Please name the category group you are looking for, i.e. banking products")
class_of_categories = col2.text_input("Insert category class:", value = 'banking products')
st.session_state['class_of_categories'] = class_of_categories

st.subheader("Categories and Products", help = "Please specify all the individual categories, products or classifications within the category group, i.e. credit products (comma-separated)")
categories = st.text_input(label = "Specify all items:", value = "FX_HEDGING, COMMODITIES_HEDGING, INTEREST_RATE_HEDGING, CREDIT, INSURANCE, FACTORING, PENSIONS, ESG, CASH_MANAGEMENT, DEPOSITS, ASSET_MANAGEMENT, OTHER")
categories = categories.split(',')
categories = [el.strip(' ') for el in categories]
st.session_state['categories'] = categories

st.subheader("Objective", help="**Attention:** Please phrase the verb in your objective in present progressive tense (-ing), such as this example: 'extracting company specific sales leads for any banking products relating to capital markets or asset management'")
objective = st.text_area(label="Describe your objective:", max_chars=200, value = "extracting company specific information that indicate sales opportunities for products relating to capital market or asset management")

"---"

# Show generated instructions to user
instructions = f"""You instruct the model to analyse {type_of_documents}s by {objective}.

The model will extract and classify information from your {type_of_documents} for the following categories of {class_of_categories}:
{', '.join(categories)}."""

st.subheader("Your generated instructions:")
st.write(instructions)

"---"

# Upload multiple txt train files 
st.subheader("Let's train our model! :weight_lifter:")

uploaded_file = st.file_uploader(
    "Upload training data (XLSX)", 
    accept_multiple_files=True, 
    type="xlsx",
    help="Please prepare your XLSX file with specific columns containing the context, categories and quote."
)

st.session_state['objective'] = objective

"---"

st.subheader("Your metric")
metric = st.radio("Please chose your metric", ["Substring Quote and Categories", "LLM Quotes and Categories"], help="Substring quote and categories measures the exact accuracy by comparing the expected and predicted quote and categories. LLM Quotes and Categories lets the LLM assess whether the predicted quote and categories are relevant in relation to the expected results.")

if metric == "Substring Quote and Categories":
    metric = 'simple'
elif metric == "LLM Quotes and Categories":
    metric = 'llm'

st.session_state['metric'] = metric

"---"

st.subheader("Your Optimizer")
optimizer = st.radio("Please choose your optimizer", ["Bootstrap Few Shot", "Bootstrap Few Shot and COPRO"], help="Bootstrap Few Shot automatically generates examples for few shot learning. COPRO calls the LLM to optimise the prompt automatically. It is recommended to start with Bootstrap Few Shot first and only run both optimizers if you have more time and resources.")

if optimizer == "Bootstrap Few Shot":
    optimizer = 'bootstrap-few-shot'

elif optimizer == "Bootsrap Few Shot and COPRO":
    optimizer = 'combined'

st.session_state['optimizer'] = optimizer


submit_button = st.button('Run optimizer')

if submit_button:
    if type_of_documents!= '' and categories!= [] and class_of_categories!= '' and objective!= '' and metric!='':
        model_path, model_number= optimize(type_of_documents, categories, class_of_categories, objective, metric, optimizer)
        st.write("You model ID is:")
        st.write(model_number)
        st.session_state['model_path'] = model_path
        st.session_state['model_number'] = model_number


