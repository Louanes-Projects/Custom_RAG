from langchain.prompts import PromptTemplate
from .config import settings
from .llm_clients import GPT_Model,Palm_Model
import pprint
from ..vectorstore.documentretriever import KnowledgeRetriever
import logging

logging.basicConfig(level=logging.DEBUG)

retriever = KnowledgeRetriever(**settings["retriever"])


def python_format(string):
    """put a string inside a Markdown code block"""
    return f"```python\n{string}\n```"


def check_template_placeholders(template_str):
    """Check if the placeholders inside the function are correct."""
    template = PromptTemplate.from_template(template_str)
    prompt_kwargs = {"user_query": "dummy string", "context": "dummy string"}
    try:
        template.format(**prompt_kwargs)
        return ""
    except KeyError as e:
        return e.args[0]


def update_template(template_str, config):
    """Update the template string used for the prompt."""
    wrong_placeholders = check_template_placeholders(template_str)
    if not wrong_placeholders:
        config["template_str"] = template_str
    display_template = f"{wrong_placeholders}\n{config['template_str']}"
    return python_format(display_template), config, ""


def user(user_message, chatbot):
    """Handle the message history."""
    logging.info("User function called.")

    return "", chatbot + [[f"{user_message}", None]]

def list_collections():
    # Placeholder function; replace this with actual logic to retrieve collections from your knowledge retriever container.
    collections = retriever.list_collections()
    list_collections = [elem["name"] for elem in collections]
    logging.debug(f'collections {list_collections}')
    return list_collections


    

def bot(history, sessionconfig, temperature, top_p, max_output_tokens, chunks_qty,ai_source,enable_retriever,selected_collection):
    """Process the user's query and generate a response."""
    history[-1][1] = ""
    user_query = "".join(history[-1][0])
    logging.info("Bot function called.")
    logging.info(f"------------------------------session_config inside bot ---- {sessionconfig}.")

    documents = {}
    try:
        if enable_retriever.value == "Enabled":
            # Retrieve documents
            documents = retriever.retrieve_documents(user_texts=[user_query], collection_name=selected_collection.value, n_results=chunks_qty)[0]
            logging.info(f"############-documents ---- {documents}")

            chunksdisplayed = [{"text": doc.page_content, "similarity": doc.metadata["distance"]} for doc in documents]
            chunksdisplayed_str = f"```\n{pprint.pformat(chunksdisplayed, indent=1, width=80, sort_dicts=False)}\n```"

        else: 
            chunksdisplayed_str = "Knowledge Retriever is not enabled"
    except:
        chunksdisplayed_str = "Knowledge Retriever is not enabled, go to the Vectorstore Tab to enable it and select a collection of documents."



    # Create prompt
    if ai_source == "AzureOpenAI":
        prompt = create_prompt_azure(history, sessionconfig, documents,enable_retriever)
        client = GPT_Model(settings["inference_server"]["azure"],sessionconfig)
        for elem in client.send_prompt(prompt):
            history[-1][1] += elem
            yield history, sessionconfig ,temperature, top_p, max_output_tokens, chunks_qty, chunksdisplayed_str, prompt
    elif ai_source == "GooglePalm":
        prompt = create_prompt_google(history, sessionconfig, documents,enable_retriever)
        client = Palm_Model(settings["inference_server"]["google"],sessionconfig)
        logging.debug(f'This is the prompt sent {prompt}')

        for elem in client.send_prompt(prompt):

            history[-1][1] += elem
            yield history, sessionconfig,temperature, top_p, max_output_tokens, chunks_qty, chunksdisplayed_str, prompt
    else:
        raise ValueError("Invalid AI source selected.")


def create_prompt_azure(history, config,documents,enable_retriever):
    # The logic for creating AzureOpenAI specific prompt
    user_query = "".join(history[-1][0])

    if enable_retriever == "Enabled":
        context = "\n - " + "\n - ".join([f'"{document.page_content}"' for document in documents])
        template = PromptTemplate.from_template(config["kr_template_str"])
    
    else:
        context =""
        template = PromptTemplate.from_template(config["default_template"])
        logging.debug(f'collections {template}')
    return template.format(user_query=user_query, context=context)

def create_prompt_google(history, config, documents,enable_retriever):
    user_query = "".join(history[-1][0])
    
    if enable_retriever == "Enabled":
        context = "\n - " + "\n - ".join([f'"{document.page_content}"' for document in documents])
        template = PromptTemplate.from_template(config["kr_template_str"])
    
    else:
        context =""
        template = PromptTemplate.from_template(config["default_template"])
        logging.debug(f'collections {template}')

    return template.format(user_query=user_query, context=context)


def process_uploaded_files(files):
    for file_info in files:
        filename = file_info["name"]
        file_data = file_info["content"]
        logging.debug(f"file name : {filename}")

        # Process each file (e.g., save to disk, parse contents, etc.)
        with open(filename, 'wb') as f:
            f.write(file_data)
    return "Files uploaded and processed successfully!"

