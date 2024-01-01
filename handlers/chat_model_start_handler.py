from langchain.callbacks.base import BaseCallbackHandler


class ChatModelStartHandler(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        # c:/Users/ERASMUS/Documents/projects/langchain/main.py
        print("Starting chat model...")
        print(serialized)
        
