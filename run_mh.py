import setup_clarifai_llm
#import setup_clarifai_llm_retriever
import multihop 
#import rag
import dspy
from doc_load_split import docs, documents
import yaml

sentence = "Fuel pump is broken"

# Define the question to ask the RAG program
my_question = "How to change the brake fluid"

# Create a RAG (Retrieval-Augmented Generation) object
multihop = multihop.MultiHop(passages_per_hop =3)
#Rag_obj = rag.RAG()

# Get the prediction from the RAG model for the given question.
# This prediction includes both the context and the answer.
predict_response_llama70b = multihop(context = documents, question = my_question)
#predict_response_llama70b = Rag_obj(my_question)

# Print the question, predicted answer, and truncated retrieved contexts.
print(f"Question: {my_question}")
print(f"Predicted Answer: {predict_response_llama70b.answer}")
#print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in predict_response_llama70b.context]}")
