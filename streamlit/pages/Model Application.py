import streamlit as st
import os
import sys
sys.path.append(os.path.dirname('/home/cdsw/CB2MUHX/project-weeks-promptu/'))
from module.azure_openai import AzureOpenAI
from io import StringIO
import httpx
import pandas as pd

import dspy
from datetime import datetime

sys.path.append(os.path.dirname('/home/cdsw/CB2MUHX/project-weeks-promptu/prompt_optimization'))

from prompt_optimization.signatures import init_signatures
from prompt_optimization.scan_report import ScanReport
from prompt_optimization.output_postprocessing import aggregate_per_category

print('------------------------------------------')
print(datetime.now())

# Streamlig config and header

st.set_page_config(page_title='Sales Leads from Annual Reports by AI', layout='wide', page_icon='assets/favicon.ico')

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('assets/style.css')

# Configure page header
col1, col2= st.columns([5,1], gap="small")
col2.image("assets/coba-logo.png", width=500)

hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
        .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
    </style>
'''

st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

col1, col2= st.columns([1, 10], gap="small")
col1.image("assets/CSA_Pyramid_Logo.png", width=100)
col2.title('Sales Leads from Annual Reports by AI')
st.markdown('#')


#st.markdown('''<div style="background:#002e3c;
#                        text-align: left;
#                        border-radius: 10px;
#                        #width: 200px;
#                        height: 50px;
#                        "> <img src='assets/coba-logo.png'> </div>''', unsafe_allow_html=True)


# Functions

def store_llm_key():
    st.session_state.llm_key = st.session_state.text_input
    st.session_state.text_input = ''
    st.session_state.button_llm_key_disabled = True

def clear_sales_lead():
    st.session_state.response = ''
    st.session_state.elapsed_time = ''
    st.cache_data.clear()

@st.cache_data
def read_report(uploaded_file):
    print('read file')

    try:
        report = ''.join(uploaded_file.getvalue().decode("utf-8"))
        st.session_state.report = report.replace('<em>', '').replace('</em>', '')

        with st.expander('Report', expanded=False):
            st.markdown(st.session_state.report)
        st.write("##")
    except Exception as e:
        st.error(f"An error occurred while reading the file: {str(e)}")
        st.session_state.report = None

    return st.session_state.report

# session_state variables
if 'text_input' not in st.session_state:
    st.session_state.text_input = ''

if 'llm_key' not in st.session_state:
    st.session_state.llm_key = ''

if 'response' not in st.session_state:
    st.session_state.response = ''

if 'report' not in st.session_state:
    st.session_state.report = ''

if 'elapsed_time' not in st.session_state:
    st.session_state.elapsed_time = ''

if 'show_button_llm_key' not in st.session_state:
    st.session_state.show_button_llm_key = True

if 'model_number' not in st.session_state:
    st.session_state.model_number = ''

if 'model_path' not in st.session_state:
    st.session_state.model_path = ''

if 'type_of_documents' not in st.session_state:
    st.session_state.type_of_documents = ''

if 'objective' not in st.session_state:
    st.session_state.objective = ''

if 'number_of_items_in_output' not in st.session_state:
    st.session_state.number_of_items_in_output = 5

if 'class_of_categories' not in st.session_state:
    st.session_state.class_of_categories = ''

if 'categories' not in st.session_state:
    st.session_state.categories = []

if 'df_response' not in st.session_state:
    st.session_state.df_response = ''


# Model selection
model_id = st.text_input('Please enter the ID of your compiled model', max_chars=10, value = st.session_state['model_number'])
model_path = f"../models/{model_id}.json"

# Input of LLM Key
llm_key = st.text_input('Please enter LLM key', key= 'text_input', type='password')

if st.session_state.show_button_llm_key:
    button = st.button('Save LLM key', on_click=store_llm_key, disabled=False)
    st.session_state.show_button_llm_key = False
else:
    button = st.button('LLM key saved', on_click=store_llm_key, disabled=True)
st.write("##")


# File upload
uploaded_file = st.file_uploader("Choose a file", type=['txt'], on_change=clear_sales_lead)  # , accept_multiple_files=False
if uploaded_file is not None:
    read_report(uploaded_file)


# Request

def make_request(llm_key_list):

    start_time = datetime.now()

    print('-------------------------------')
    print('-------------------------------')
    print('-------------------------------')


    llm_key = llm_key_list

    print(f'In function make_request, llm_key= {llm_key}')

    # environment
    environment = 'TUD'

    # Choose model
    #llm_model_name = 'gpt-4-reproducible'
    llm_model_name = 'gpt-4o'
    #llm_model_name = 'gpt-4-32k'
    #llm_model_name = 'gpt-35-turbo'
    #llm_model_name = 'gpt-35-turbo-16k'

    headers={'llm-key': llm_key}

    prompt_user = st.session_state.report

    print(prompt_user)

    # llm credentials
    lm_gpt = AzureOpenAI(
        tud_dev = environment,
        api_version = '2024-06-01', #'2024-06-01',#'2023-07-01-preview',
        model_name = llm_model_name, 
        api_key = llm_key,
        model_type = "chat"
    )

    dspy.settings.configure(lm=lm_gpt)

    type_of_documents = st.session_state['type_of_documents']
    number_of_items_in_output = 5
    objective = st.session_state['objective']
    class_of_categories = st.session_state['class_of_categories']
    categories = st.session_state['categories']

    hint, ChunkerSignature, PredictRelevance, PredictCategory, Translator = init_signatures(type_of_documents,number_of_items_in_output,objective, class_of_categories, categories)
    scan = ScanReport(hint, ChunkerSignature, PredictRelevance, PredictCategory, Translator)

    scan.load(path=model_path)

    result = scan(prompt_user)

    result_dict = aggregate_per_category(result.answer, categories)
    
    # st.write(result_dict)
    
    st.session_state.response = result_dict

    df = pd.DataFrame.from_dict(result_dict)
    df = df.transpose()
    st.session_state.df_response = df

    elapsed_time = datetime.now() - start_time
    st.session_state.elapsed_time = f'elapsed time: {round(elapsed_time.seconds / 60, 1)} minutes'
    print(st.session_state.elapsed_time)



button = st.button('Get Sales Leads', on_click=make_request, args=[st.session_state.llm_key])
st.write(st.session_state.df_response)
st.write(st.session_state.response)
st.write("##")
st.write(st.session_state.elapsed_time)
st.write("#")