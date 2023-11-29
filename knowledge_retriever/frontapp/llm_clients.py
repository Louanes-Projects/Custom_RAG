import requests
import openai
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import HumanMessage
from langchain import PromptTemplate, LLMChain
from abc import ABC, abstractmethod
from vertexai.preview.language_models import ChatModel, InputOutputTextPair
from .config import settings
from typing import Optional
from google.auth import credentials as auth_credentials



settings_google =settings["inference_server"]["google"]
settings_azure = settings["inference_server"]["azure"]
class LLMBase(ABC):
    DEFAULT_CONFIG = {
        "model": "xxx",
        "base": "http://20.23.124.219/v1",
        "min_tokens": 0,
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "stream": True
    }

    def __init__(self, settings, config=None):
        self._settings=settings
        self.begining_sentence = None
        self.end_of_text = None
        self.config = config or self.DEFAULT_CONFIG

    @abstractmethod
    def send_prompt(self, system_message , question):
        raise NotImplementedError("send_prompt method is not implemented")


class GPT_Model(LLMBase):
    def __init__(self, _settings, sessionconfig=None):
        self._settings = settings_azure
        super().__init__(_settings, sessionconfig)
    
    def send_prompt(self, question):
        try:
            llm = AzureChatOpenAI(
                                 deployment_name = self._settings["model"],
                                 openai_api_base = self._settings["api_base"],
                                 openai_api_key = self._settings["api_key"],
                                 openai_api_version = self._settings["api_version"],
                                 temperature = self.config["temperature"],
                                 max_tokens = self.config["max_tokens"],
                                 streaming=True
                              )
            result = llm([HumanMessage(content=question)]).content
            yield result
        except requests.exceptions.RequestException as e:
            print(f"Error: Unable to send request: {str(e)}")
    
class Palm_Model(LLMBase):
    def __init__(self, _settings, config=None):
        self._settings = settings_google
        super().__init__(_settings, config)

    
    def send_prompt(self,body):
        chat_model = ChatModel.from_pretrained(self._settings["model"])
        parameters = {
            "temperature": self.config["temperature"],
            "max_output_tokens": self.config["max_tokens"],
            "top_p": self.config["top_p"],
            "top_k": self.config["top_k"]
        }

        chat = chat_model.start_chat(
            context=body,
            examples=[
                InputOutputTextPair(
                    input_text="How many moons does Mars have?",
                    output_text="The planet Mars has two moons, Phobos and Deimos.",
                ),
            ],
        )
        response = chat.send_message(body, **parameters)
        return response.text


# Vertex AI SDK Initialization

def init_sample(
    project: Optional[str] = None,
    location: Optional[str] = None,
    experiment: Optional[str] = None,
    staging_bucket: Optional[str] = None,
    credentials: Optional[auth_credentials.Credentials] = None,
    encryption_spec_key_name: Optional[str] = None,
):

    from google.cloud import aiplatform

    aiplatform.init(
        project=project,
        location=location,
        experiment=experiment,
        staging_bucket=staging_bucket,
        credentials=credentials,
        encryption_spec_key_name=encryption_spec_key_name,
    )

# Call the initialization function with appropriate arguments
init_sample(project=settings_google["project"], location=settings_google["location"])


