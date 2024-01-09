import re

from openai import OpenAI

import tiktoken

ANSWER_REGEX = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)


class LLM:
    """
    The `LLM` class represents a large language model used for generating answers based on prompts.

    Attributes:
        - client: The client object used for making API requests (OpenAI compatible).
        - model_name: The name of the language model.
        - max_answer_tokens: The maximum number of tokens to generate in the answer.
        - use_claude_fix: A boolean indicating whether to use the fix for better results in Anthropic models.
        - encoding: The tokenizer used for the language model.

    Methods:
        - __init__(self, client, model_name, max_answer_tokens): Initializes the LLM object with the specified arguments.
        - claude_prompt_fix(self, prompt): Fixes the prompt for better results in Anthropic models.
        - answer(self, prompt, output_json=False): Generates an answer based on the prompt.
    """

    def __init__(self, client: OpenAI, model_name, max_answer_tokens):
        self.client = client
        self.model_name = model_name
        self.max_answer_tokens = max_answer_tokens
        self.use_claude_fix = "claude" in model_name or "pulze" in model_name

        try:
            self.encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            self.encoding = tiktoken.encoding_for_model("gpt-4")

    def claude_prompt_fix(self, prompt):
        """This seems to give better results for Anthropic models"""
        return (
            prompt
            if not self.use_claude_fix
            else f"""


    Human:
    {prompt}


    Please output your answer within <answer></answer> tags.


    Assistant: <answer>"""
        )

    def answer(self, prompt, output_json: bool = False, **kwargs):
        """Ask LLM and parse the answer.

        :param prompt: The prompt for generating the answer.
        :param output_json: A boolean indicating whether the response should be returned as JSON. Default is False.
        :return: The generated answer.
        """
        response_content = (
            self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": self.claude_prompt_fix(prompt),
                    }
                ],
                model=self.model_name,
                # This parameter is not supported by Pulze
                response_format={"type": "json_object" if output_json else "text"},
                max_tokens=self.max_answer_tokens,
                **kwargs,
            )
            .choices[0]
            .message.content
        )

        # Sometimes we get "bla bla bla <answer>good stuff</answer> bla bla bla"
        # Sometimes we get "bla bla bla: good stuff</answer>"
        if "<answer>" not in response_content:
            return response_content.removesuffix("</answer>")
        return ANSWER_REGEX.search(response_content).group(1)
