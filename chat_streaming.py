from typing import Any, Dict, List, Optional
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text

    # def on_retriever_start(self):
    #     pass

    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text+=token
        self.container.markdown(self.text) 