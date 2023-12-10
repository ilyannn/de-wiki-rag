import re
import tiktoken


class LLM:
    def __init__(self, client, model_name, max_answer_tokens):
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

    def answer(self, prompt, output_json: bool = False):
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
                response_format=("json_object" if output_json else "text"),
                max_tokens=self.max_answer_tokens,
            )
            .choices[0]
            .message.content
        )

        # Sometimes we get "bla bla bla <answer>good stuff</answer> bla bla bla"
        # Sometimes we get "bla bla bla: good stuff</answer>"
        if "<answer>" not in response_content:
            return response_content.removesuffix("</answer>")
        return re.search(r"<answer>(.*?)</answer>", response_content, ).group(1)
