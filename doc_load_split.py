# Import necessary modules from LangChain for text splitting, document loading, and Clarifai vector storage
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Clarifai as clarifaivectorstore
import yaml

# Reading YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
USER_ID=config["clarifai-vars"]["user_id"]  # User ID for authentication
APP_ID=config["clarifai-vars"]["app_id"]   # App ID for authentication
PAT=config["clarifai-vars"]["pat"] 
MODEL_URL = config["clarifai-vars"]["model_url"]
DATAFILE = config["datafile"]
print(USER_ID)

# Initialize a TextLoader object with the path to the text file containing documents to ingest
loader = TextLoader(DATAFILE) # Replace with your file path

# Load documents using the loader
documents = loader.load()

# Set up a CharacterTextSplitter to split documents into smaller chunks for efficient processing
text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=200)

# Split documents into smaller chunks
docs = text_splitter.split_documents(documents)