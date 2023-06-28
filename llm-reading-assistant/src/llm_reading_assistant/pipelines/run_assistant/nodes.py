"""
This is a boilerplate pipeline 'run_assistant'
generated using Kedro 0.18.10
"""
import logging
from typing import Literal

from .requests import (
    AzureOpenAIAPIRequest,
    OpenAIAPIRequest,
    VertexAIPaLMAPIRequest,
)

logger = logging.getLogger(__name__)


def complete_request(
    api: Literal["OpenAI", "VertexAI PaLM"],
    mode: Literal["explain", "summarize"],
    input_text: str,
    instructions: dict[str],
    max_tokens: dict[int],
    model: str,
) -> str:
    """Execute prompt and return text answer from model.

    Args:
        api (Literal[&quot;openai&quot;, &quot;vertexai_palm&quot;]): API to be used to process request
        mode (Literal[&quot;explain&quot;, &quot;summarize&quot;]): execution mode; either `explain` or `summarize`
        input_text (str): text to be explained or summarized
        instructions (dict[str]): additional instructions for the model depending on the `mode`
        max_tokens (dict[int]): maximum number of tokens to generate depending on the `mode`
        model (str, optional): OpenAI model name.

    Returns:
        str: text answer from model
    """
    logger.info("Selected API: " + api)
    logger.info("Selected model: " + model)
    logger.info("Instruction: " + instructions[mode])
    logger.info("Input: " + input_text)
    logger.info("Executing prompt...")

    if api == "OpenAI":
        APIRequest = OpenAIAPIRequest
    elif api == "VertexAI PaLM":
        APIRequest = VertexAIPaLMAPIRequest
    else:
        APIRequest = AzureOpenAIAPIRequest

    request = APIRequest(
        mode=mode,
        input_text=input_text,
        instructions=instructions,
        max_tokens=max_tokens,
        model=model,
    )
    request.execute_prompt()
    answer = request.extract_answer()

    logger.info("Model's answer:\n" + answer)
    logger.info("Done.")

    return answer
