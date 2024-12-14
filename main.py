import os
from dotenv import load_dotenv, set_key
from ai_conversation import AIConversation


def main():

    # Load environment variables from the .env file
    load_dotenv()

    # Retrieve configuration from environment variables
    models = {
        1: os.getenv("MODEL_1"),
        2: os.getenv("MODEL_2")
    }
    ollama_endpoint = os.getenv("OLLAMA_ENDPOINT")
    initial_prompt = os.getenv("INITIAL_PROMPT", "")
    max_tokens = int(os.getenv("MAX_TOKENS", 4000))
    print(f"Max tokens: {max_tokens or 'No limit!'}")

    # Initialize the AI conversation object and start it
    conversation = AIConversation(models, ollama_endpoint, max_tokens)
    conversation.start_conversation(initial_prompt)
    conversation.save_conversation_log()


if __name__ == "__main__":
    main()
