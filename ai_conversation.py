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
            0: [],
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

    def trim_messages(self, messages):

        is_limit_reached = False
        token_count = 0

        # Trim old messages to stay within the token limit
        for i, message in enumerate(messages):
            if i < 2:
                continue  # Avoid trimming system prompt and initial message

            token_count = self.count_tokens(messages)

            if not self.limit_tokens or token_count < self.max_tokens:
                print(colored(f"[SYSTEM] Context token count: {token_count} ({i-2} messages trimmed)", "magenta"))
                return messages

            if not is_limit_reached:
                print(colored(f"[SYSTEM] Max tokens reached ({token_count} of {self.max_tokens})", "magenta"))
                is_limit_reached = True

            message_count = len(messages)

            if i >= message_count - 1 and not message_count <= 2:
                break  # Avoid trimming most recent messages, unless it's the only one

            lines = message["content"].splitlines()
            first_paragraph = ""

            if len(lines) >= 1:
                first_paragraph = lines[0]

            if first_paragraph and len(first_paragraph) < len(message["content"]):
                print(colored(f"[SYSTEM] Trimming message #{i} [{messages[i]['content'][:10]}...] to one paragraph...", "magenta"))
                messages[i]["content"] = first_paragraph
                continue

            for split_token in [".", "!", "?", ":", "-", ")", "(", "*"]:
                first_sentance = first_paragraph.split(split_token)[0]

                if first_sentance and len(first_sentance) < len(first_paragraph):
                    print(colored(f"[SYSTEM] Trimming message #{i} [{messages[i]['content'][:10]}...] to one sentance{split_token}..", "magenta"))
                    messages[i]["content"] = first_sentance
                    break

        # Remove messages from the beginning until we're under the token limit
        for i, message in enumerate(messages):
            if i < 2:
                continue  # Avoid removing system prompt and initial message

            if not message["content"]:
                continue  # Message already erased

            token_count = self.count_tokens(messages)

            if not self.limit_tokens or token_count < self.max_tokens:
                print(colored(f"[SYSTEM] Context token count: {token_count} ({i-2} messages erased)", "magenta"))
                break

            print(colored(f"[SYSTEM] Erasing message #{i} [{messages[i]['content'][:10]}...]", "magenta"))
            messages[i]["content"] = "" # Completely erase the old message

        return messages

    def start_conversation(self, initial_message, num_exchanges=0, options=None):
        
        # Main conversation loop
        response_content = None
        interrupt = True  # Prompt for user message if needed
        active_ai = 0  # Starting with initial_message if set
        messages = self.messages[2]
        i = 1

        while num_exchanges == 0 or i < num_exchanges:
            try:
                if interrupt:
                    print(colored("\nPress CTRL+C to interrupt the conversation.", "red"))

                    if not response_content:
                        display_name = (f"#{i} {'Interruption' if active_ai else 'Initial message'} "
                                        f"({'SYSTEM' if initial_message else 'USER'}): ")

                    if not active_ai and initial_message:
                        print(colored(display_name + initial_message, COLOR[active_ai]))
                        response_content = initial_message

                    if not response_content:
                        print(colored("Say something or type '/bye' to stop...", "red"))
                        response_content = input(colored(display_name, COLOR[active_ai]))

                    if response_content.rstrip().endswith("/bye"):
                        print(colored("Conversation ended by USER.", COLOR[active_ai]))
                        break

                    interrupt = False

                # Switch to the other AI for the next turn
                active_ai = 2 if active_ai == 1 else 1
                i += 1

                # Add previous message to conversation histories
                messages.append({"role": "assistant", "content": response_content})
                messages = self.messages[active_ai]
                messages.append({"role": "user", "content": response_content})
                self.conversation_log.append(display_name + response_content)

                # Trim messages and get token count
                messages = self.trim_messages(messages)
                display_name = f"#{i} {self.models[active_ai].upper()} (AI {active_ai}): "
                response_content = ""

                # Generate AI response
                stream = self.client.chat(
                    model=self.models[active_ai],
                    messages=messages,
                    options=options,
                    stream=True,
                )

                for n, chunk in enumerate(stream):
                    if not n:
                        # Print the model name
                        print("\n" + colored(display_name, COLOR[active_ai]), end='', flush=True)

                    elif chunk["done"]:
                        # Print token count of this response
                        print(colored(f"[+{n}]", "magenta"))
                        break

                    # Print AI response as it streams
                    print(colored(chunk["message"]["content"], COLOR[active_ai]), end='', flush=True)
                    response_content += chunk["message"]["content"]

                # Check for conversation end condition
                if response_content.rstrip().endswith("{{end_conversation}}"):
                    print(colored(f"Conversation ended by AI {active_ai}.", COLOR[active_ai]))
                    break

            except KeyboardInterrupt:
                if interrupt:
                    break

                interrupt = True

        print(colored(f"\nConversation ended after {i-1} messages.", "red"))

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
