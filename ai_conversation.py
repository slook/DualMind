import ollama
import json
from termcolor import colored
import datetime
import tiktoken  # Used for token counting

COLOR = {0: "yellow", 1: "cyan", 2: "green"}


class Loader:

    @staticmethod
    def get_options(filename="options.json"):
        """Load options from a JSON file."""
        with open(filename, "r") as file:
            options = json.load(file)
        return options

    @staticmethod
    def get_system_prompt(active_ai, old_system_prompt):
        """Load the system prompt from a text file."""

        with open(f"system_prompt_{active_ai}.txt", "r") as file:
            new_system_prompt = file.read().strip(" \n")

        if new_system_prompt != old_system_prompt:
            print(colored(f"[SYSTEM] AI {active_ai} system prompt updated: "
                          f"{new_system_prompt.count(' ')} words.", "magenta"))

        return {"role": "system", "content": new_system_prompt}

loader = Loader()
   

class AIConversation:

    def __init__(self, models, ollama_endpoint, max_tokens):

        self.models = models
        self.ollama_endpoint = ollama_endpoint
        self.max_tokens = max_tokens

        self.client = ollama.Client(ollama_endpoint)
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

        self.messages = {1: [], 2: []}
        self.conversation_log = []

    def count_tokens(self, messages):
        # Count the total number of tokens in the messages
        return sum(len(self.tokenizer.encode(msg["content"])) for msg in messages)

    def trim_messages(self, active_ai):

        messages = self.messages[active_ai]
        token_count = self.count_tokens(messages)

        # Remove pairs of messages from the beginning until we're under the token limit
        while self.max_tokens and len(messages) > 3 and token_count > self.max_tokens:
            print(colored(
                f"[SYSTEM] Removed oldest {messages[2]['role']} [{messages[2]['content'][:10]}...]"
                f" & {messages[3]['role']} [{messages[3]['content'][:10]}...] exchange.", "magenta"
            ))
            # Avoid trimming system prompt [0], and initial message [1]
            del messages[3]
            del messages[2]
            token_count = self.count_tokens(messages)

        print(colored(f"[SYSTEM] AI {active_ai} context: {token_count} tokens in {len(messages)} messages.", "magenta"))

    def start_conversation(self, initial_message):

        response_content = None
        interrupt = True  # Prompt for user message if needed
        active_ai = 0
        i = 1

        # Main conversation loop
        while True:
            try:
                if interrupt:
                    print(colored(f"\nPress CTRL+C to interrupt the conversation.", "red"))

                    if not response_content:
                        model_name = "Interruption " if active_ai else "Initial message "

                        if not active_ai and initial_message:
                            model_name += "(ENV)"
                            print(f"#{i} {colored(model_name, COLOR[active_ai])}: {initial_message}")
                            response_content = initial_message
                        else:
                            model_name += "(USER)"
                            response_content = input(f"#{i} {colored(model_name, COLOR[active_ai])}: ")

                    if not active_ai:
                        active_ai = len(self.models)

                    interrupt = False

                # Add previous message to conversation histories
                self.conversation_log.append(model_name + response_content)
                self.messages[active_ai].append({"role": "assistant", "content": response_content})

                if response_content.rstrip().endswith("{{end_conversation}}"):
                    break

                # Switch to the next AI in turn
                active_ai = (active_ai + 1) if active_ai < len(self.models) else 1

                self.messages[active_ai].append({"role": "user", "content": response_content})
                self.messages[active_ai][0] = loader.get_system_prompt(active_ai,
                                                                       self.messages[active_ai][0].get("content", ""))
                self.trim_messages(active_ai)  # and print token count

                i += 1
                options = loader.get_options()
                options["seed"] = i  # Reproducable outputs
                model_name = f"{self.models[active_ai].upper()} (AI {active_ai})"
                response_content = ""

                # Generate AI response
                stream = self.client.chat(
                    model=self.models[active_ai],
                    messages=self.messages[active_ai],
                    options=options,
                    stream=True,
                )

                # Print AI response while it streams
                for t, chunk in enumerate(stream):
                    if chunk["done"]:
                        print(colored(f"[+{t}]", "magenta"))
                        break

                    if not t:
                        print(f"\n#{i} {colored(model_name, COLOR[active_ai])}: ", end='', flush=True)

                    print(colored(chunk["message"]["content"], COLOR[active_ai]), end='', flush=True)
                    response_content += chunk["message"]["content"]

            except ollama.ResponseError as err:
                print(colored(f"[SYSTEM] {err}", "red"))
                response_content += f"\n[SYSTEM] {err}"

            except KeyboardInterrupt:
                if interrupt:
                    break

                interrupt = True

        print(colored(
            f"\nConversation ended by {'USER' if interrupt else 'AI'} after {i-1} messages.",
            COLOR[0 if interrupt else active_ai]
        ))

    def save_conversation_log(self, filename=None):

        if len(self.conversation_log) < 2:
            return

        # Save the conversation log to a file
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_log_{timestamp}.txt"

        log_content = f"Conversation Log - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        log_content += f"Ollama Endpoint: {self.ollama_endpoint}\n"
        log_content += f"Model 1: {self.models[1]}\n"
        log_content += f"Model 2: {self.models[2]}\n"
        log_content += f"System Prompt 1:\n{self.messages[1][0]['content']}\n\n"
        log_content += f"System Prompt 2:\n{self.messages[2][0]['content']}\n\n"
        log_content += "Conversation:\n\n"

        for message in self.conversation_log:
            log_content += f"{message}\n\n"

        try:
            with open(filename, "w") as f:
                f.write(log_content)
        except:
            print(colored(f"Cannot save conversation log to {filename}", "red"))
            return

        print(f"Conversation log saved to {filename}")
