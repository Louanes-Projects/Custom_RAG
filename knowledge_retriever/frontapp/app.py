from gradio import components
import gradio as gr
from .config import settings
from langchain.prompts import PromptTemplate
from . import utils
import logging
import time
logging.basicConfig(level=logging.DEBUG)
time.sleep(5)



def handle_dropdown_change(selected_value):

    selected_collection = gr.State(value=selected_value)
    logging.info(f"The selected collection .value in the handle change function is: {selected_collection.value}")
    return selected_collection

def handle_retriever(selected_value):

    enable_retriever_selection = gr.State(value=selected_value)
    logging.info(f"The selected enable_retriever_selection .value in the handle change function is: {enable_retriever_selection.value}")
    return enable_retriever_selection


# Constants and Initializations
default_config = settings["inference_server"]["llm_config"].to_dict()

collections_state = gr.State(value=utils.list_collections())





# Gradio Interface
with gr.Blocks() as app:
 
    selected_collection = gr.State(value="")
    enable_retriever_selection=gr.State(value="Disabled")
    logos = gr.Markdown("<div style='display: flex; align-items: center; margin-bottom: 20px !important; margin-top: 20px; justify-content: center;'><img src='file/knowledge_retriever/frontapp/static/Inetum2.png' width='25%' height='25%' style='float: left;'></div>") 
    
    with gr.Tabs():

        

        with gr.Tab("Chat"):
            with gr.Row():
                
                with gr.Column(scale=8):
                    chatbot = gr.Chatbot(show_label=False, height=600)
                
                with gr.Column(scale=4):
                    ai_source = gr.Dropdown(choices=['AzureOpenAI', 'GooglePalm'], label="AI Source", value ="GooglePalm")
                    with gr.Accordion("Parameters", open=False):
                        temperature = gr.Slider(minimum=0.0, maximum=1.0, value=default_config["temperature"], step=0.1, interactive=True, label="Temperature")
                        top_p = gr.Slider(minimum=0.0, maximum=1.0, value=default_config["top_p"], step=0.1, interactive=True, label="Top P")
                        max_output_tokens = gr.Slider(minimum=16, maximum=1024, value=default_config["max_tokens"], step=64, interactive=True, label="Max output tokens")
                        chunks_qty = gr.Slider(minimum=0, maximum=10, value=default_config["chunks_qty"], step=1, interactive=True, label="Context chunks quantity")

            with gr.Row(visible=True):
                user_message = gr.Textbox(show_label=False, placeholder="Posez votre question et appuyez sur entrer....",container=False)
                clear = gr.Button("Clear", variant="stop")
            
            with gr.Accordion("Chunks", open=False):
                chunks_elem = gr.Markdown()


            session_config = gr.State(value=default_config)
            logging.info(f"------------------------------selcted collection value before then bot {selected_collection.value}.")
            user_message.submit(utils.user, [user_message, chatbot], [user_message, chatbot], queue=False).then(utils.bot, [chatbot, session_config, temperature, top_p, max_output_tokens, chunks_qty, ai_source,enable_retriever_selection, selected_collection], [chatbot, session_config, temperature, top_p, max_output_tokens, chunks_qty, chunks_elem])
            clear.click(lambda: None, None, chatbot, queue=False)    
        
        with gr.Tab("Vectorstore"):
            with gr.Accordion("Vectorstore Settings", open=True):
                enable_retriever = gr.Dropdown(choices=["Enabled", "Disabled"], label="Knowledge Retriever",value="Disabled")
                enable_retriever.input(handle_retriever, inputs=enable_retriever,outputs=enable_retriever_selection)

                collection_selection = gr.Dropdown(choices=collections_state.value, label="Select Collection")
                collection_selection.input(handle_dropdown_change, inputs=collection_selection,outputs=selected_collection)
                file_upload = gr.Files(type="file", label="Upload Documents")
                submit_button = gr.Button("Upload")
                submit_button.click(utils.process_uploaded_files, inputs=[file_upload])
       
  
    

app.title = "IAG 3M"
app.queue()
app.launch(server_name="0.0.0.0", server_port=7000, favicon_path="./knowledge_retriever/frontapp/static/favicon.png") 
