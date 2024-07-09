# Import necessary modules from LangChain for text splitting, document loading, and Clarifai vector storage
from doc_load_split import docs
import yaml
# Importing necessary modules
import dspy
from dspy.retrieve.clarifai_rm import ClarifaiRM 

# Reading YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
USER_ID=config["clarifai-vars"]["user_id"]  # User ID for authentication
APP_ID=config["clarifai-vars"]["app_id"]   # App ID for authentication
PAT=config["clarifai-vars"]["pat"] 
MODEL_URL = config["clarifai-vars"]["model_url"]
DATAFILE = config["datafile"]
print(USER_ID)

# Create a vector database using the Clarifai vector store
clarifai_vector_db = clarifaivectorstore.from_documents(
    user_id=USER_ID,  # User ID for authentication
    app_id=APP_ID,    # App ID for authentication
    documents=docs,   # Split documents
    pat=PAT           # Personal Access Token (PAT) for authentication
)


# Initializing the language model (LLM) with Clarifai
llm = dspy.Clarifai(
    model=MODEL_URL,             # Clarifai model URL
    api_key=PAT,                 # Personal Access Token (PAT) for authentication
    n=2,                         # Number of results to return
    inference_params={           # Parameters for inference
        "max_tokens": 100,       # Maximum number of tokens per input
        'temperature': 0.6       # Temperature parameter for text generation
    }
)

# Initializing the retriever model with Clarifai for document retrieval
retriever_model = ClarifaiRM(
    clarifai_user_id=USER_ID,    # User ID for Clarifai authentication
    clarfiai_app_id=APP_ID,      # App ID for Clarifai authentication
    clarifai_pat=PAT,            # PAT for Clarifai authentication
    k=2                           # Number of documents to retrieve
)

# Configuring settings for DSPy
dspy.settings.configure(
    lm=llm,                      # Language model
    rm=retriever_model           # Retriever model
)
