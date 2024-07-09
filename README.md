# RAG with DSPy and Clarifai

This project demonstrates how to build a Retrieval-Augmented Generation (RAG) system using DSPy and the Clarifai Python SDK. 

## Features
- Utilizes Clarifai's LLM models and vector database for efficient information retrieval and text generation
- Implements a RAG system that combines retrieval and generation capabilities
- Uses DSPy for streamlined development of NLP tasks

## Setup
1. Install required packages: clarifai, langchain, dspy-ai
2. Configure Clarifai credentials (PAT, User ID, App ID)
3. Ingest data into Clarifai's vector database
4. Initialize DSPy with Clarifai LLM model

## Usage
- Define custom signatures and modules for specific NLP tasks
- Use the RAG class to retrieve relevant context and generate answers based on user queries

This project showcases the integration of Clarifai's powerful AI capabilities with DSPy's flexible programming model for building advanced NLP applications.

Compiling the RAG program
Having defined this program, let's now compile it. Compiling a program will update the parameters stored in each module. In our setting, this is primarily in the form of collecting and selecting good demonstrations for inclusion in your prompt(s).

Compiling depends on three things:

A training set. We'll just use our 20 question–answer examples from trainset above.
A metric for validation. We'll define a quick validate_context_and_answer that checks that the predicted answer is correct. It'll also check that the retrieved context does actually contain that answer.
A specific teleprompter. The DSPy compiler includes a number of teleprompters that can optimize your programs.
Teleprompters: Teleprompters are powerful optimizers that can take any program and learn to bootstrap and select effective prompts for its modules. Hence the name, which means "prompting at a distance".

Different teleprompters offer various tradeoffs in terms of how much they optimize cost versus quality, etc. We will use a simple default BootstrapFewShot in this notebook.

DSPy typically requires very minimal labeling. Whereas your pipeline may involve six or seven complex steps, you only need labels for the initial question and the final answer. 
DSPy will bootstrap any intermediate labels needed to support your pipeline. If you change your pipeline in any way, the data bootstrapped will change accordingly!