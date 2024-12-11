import ollama
from termcolor import colored
import datetime
import tiktoken  # Used for token counting

COLOR = {0: "yellow", 1: "cyan", 2: "green"}

class AIConversation:
    def __init__(
        self,
        model_1,
        model_2,
        system_prompt_1,
        system_prompt_2,
        ollama_endpoint,
        max_tokens=4000,
        limit_tokens=True
    ):
        # Initialize conversation parameters and Ollama client
        self.models = {1: model_1, 2: model_2}
        self.messages = {
            #0: [],
            1: [{"role": "system", "content": system_prompt_1}],
            2: [{"role": "system", "content": system_prompt_2}]
        }
        self.client = ollama.Client(ollama_endpoint)
        self.ollama_endpoint = ollama_endpoint
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.max_tokens = max_tokens
        self.limit_tokens = limit_tokens

        self.conversation_log = []

    def count_tokens(self, messages):
        # Count the total number of tokens in the messages
        return sum(len(self.tokenizer.encode(msg["content"])) for msg in messages)

    def trim_messages(self, active_ai):

        messages = self.messages[active_ai]
        token_count = self.count_tokens(messages)

        # Remove pairs of messages from the beginning until we're under the token limit
        while self.limit_tokens and len(messages) > 3 and token_count > self.max_tokens:
            print(colored(
                f"[SYSTEM] Removed oldest {messages[2]['role']} [{messages[2]['content'][:10]}...]"
                f" and {messages[3]['role']} [{messages[3]['content'][:10]}...] messages.", "magenta"
            ))
            # Avoid trimming system prompt [0], and initial message [1]
            del messages[3]
            del messages[2]
            token_count = self.count_tokens(messages)

        print(colored(f"[SYSTEM] Context: {token_count} tokens in {len(messages)} messages.", "magenta"))

    def start_conversation(self, initial_message, num_exchanges=0, options=None):
        
        response_content = None
        interrupt = True  # Prompt for user message if needed
        active_ai = 0
        i = 1

        # Main conversation loop
        while num_exchanges == 0 or i < num_exchanges:
            try:
                if interrupt:
                    print(colored(f"\nPress CTRL+C to interrupt the conversation.", "red"))

                    if not response_content:
                        model_name = (f"#{i} {'Interruption' if active_ai else 'Initial message'} "
                                      f"({'ENV' if initial_message and not active_ai else 'USER'}): ")

                        if initial_message and not active_ai:
                            print(colored(model_name + initial_message, COLOR[active_ai]))
                            response_content = initial_message
                        else:
                            response_content = input(colored(model_name, COLOR[active_ai]))

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
                self.trim_messages(active_ai)  # and print token count

                i += 1
                model_name = f"#{i} {self.models[active_ai].upper()} (AI {active_ai}): "
                options["seed"] = i
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
                        print(colored(f"[+{t} tokens]", "magenta"))
                        break

                    if not t:
                        print(f"\n{colored(model_name, COLOR[active_ai])}", end='', flush=True)

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
