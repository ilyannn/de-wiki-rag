#!/usr/bin/env python
"""Generate context using German Wikipedia articles.

We take the top 10% paragraphs (ranked by article views), embed them using a large 
multilingual model, and cache the embeddings.
For each question, we retrieve the pieces of context and rescore them using Pulze API.
We demonstrate what the answers look with and without using the context.
 """
import json
import logging
import os
import random
from datetime import datetime

import dotenv
from langfuse import Langfuse
from langfuse.openai import openai
from datasets import load_dataset, load_from_disk
from openai import OpenAI
from txtai import Embeddings

from llm import LLM

INITIAL_QUESTIONS = [
    "How many wives can a man have in Germany?",
    "Wer ist ein Schöffe bzw eine Schöffin?",
    "Was waren die deutsch-französischen Beziehungen im 19. Jhd?",
    "Why was the Berlin wall built?",
    "Wer ist das aktuelle Staatsoberhaupt in Deutschland?",
]

DATASET_SOURCE = "Cohere/wikipedia-22-12"
DATASET_PATH = "data/de-wiki-22-12-cohere-by-views"

# much better than the default one for German text
EMBEDDINGS_MODEL = "intfloat/multilingual-e5-large"
EMBEDDINGS_HOW_MANY_K = 1500  # note the total size of the dataset is 15m embeddings
EMBEDDINGS_PATH = f"data/de-wiki-multilingual-e5-large-top-{EMBEDDINGS_HOW_MANY_K}k"

CONTEXT_CHOICES = 20

MODEL = "gpt-4-1106-preview"
MODEL_CONTEXT_LENGTH = 8192
MAX_ANSWER_TOKENS = min(4096, MODEL_CONTEXT_LENGTH)


class Corpus:
    def __init__(
        self,
        data: dict[int : dict[str:str]],
        embeddings: Embeddings,
    ):
        self.data = data
        self.embeddings = embeddings

    def format_chunk(self, chunk_id):
        return f"""{chunk_id} [{self.data[chunk_id]["title"]}] {self.data[chunk_id]["text"]}"""


def load_corpus() -> Corpus:
    """Load and cache the dataset and its embeddings."""
    try:
        data = load_from_disk(DATASET_PATH, keep_in_memory=True)
        logging.info(f"Loaded data of shape {data.shape} from {DATASET_PATH}")
    except FileNotFoundError:
        original_data = load_dataset(DATASET_SOURCE, "de", split="train")
        logging.info(f"Loaded full wikipedia data of shape {original_data.shape}")
        data = original_data.sort(["views"], reverse=True, kind="stable")
        os.makedirs(DATASET_PATH, exist_ok=True)
        data.save_to_disk(DATASET_PATH)
        logging.info(f"Saved data of shape {data.shape} to {DATASET_PATH}")

    embeddings = Embeddings(path=EMBEDDINGS_MODEL)
    logging.info(f"Loaded embedding model {EMBEDDINGS_MODEL}")

    try:
        embeddings.load(EMBEDDINGS_PATH)
        logging.info(f"Loaded {embeddings.count()} embeddings from {EMBEDDINGS_PATH}")
    except FileNotFoundError:
        top_data = next(data.iter(EMBEDDINGS_HOW_MANY_K * 1000))
        embeddings.index(top_data["text"])
        logging.debug(f"Indexed following pages: %s", set(top_data["title"]))
        embeddings.save(EMBEDDINGS_PATH)
        logging.info(f"Saved {embeddings.count()} embeddings to {EMBEDDINGS_PATH}")

    return Corpus(data, embeddings)


def build_context(context_chunks):
    """Prepare a context string out of the suggested content chunks"""
    return "\n".join(
        f"""{c["id"]} (from '{c["title"]}'): {c["text"]}""" for c in context_chunks
    )


def context_rescoring_prompt(query, context_chunks):
    """Prepare a rescoring prompt for context chunks"""
    return f"""
You are part of a text retrieval engine for German language. Your goal is to check whether the context, retrieved from the vector database, is helpful when answering the query asked.

The query: {query}

Context pieces, taken from Wikipedia articles, that you need to check:
 {build_context(context_chunks)}
 
Provide the list of ids of context pieces that help answer the question posed, in the JSON format. Do not give any other output. Do not add any ticks or other symbols around JSON. Example output: 
[76, 23, 32344123]"""


def question_prompt(query, context_string=None):
    """Prepare a question prompt that optionally includes a context"""
    context_query = (
        ""
        if not context_string
        else f"""
The following context pieces, taken from recent Wikipedia articles, might be helpful in the answer:
{context_string}

"""
    )

    return f"""You are a question-answer engine who takes great care to provide the most accurate answer. 
Answer the following question in German to the best of your ability: {query}
Aim at several paragraphs that show clear and reasoned thinking. 
{context_query}
"""


def get_context_ids(
    question: str,
    corpus: Corpus,
    llm: LLM,
    trace,
) -> (list[int], list[int]):
    """
    :param question: The question for which we want to find the context.
    :param corpus: Corpus within which we look for context.
    :param llm: The language model abstraction used for completion.
    :param trace: Langfuse trace object for observation purposes
    :return: A tuple containing suggested context IDs and IDs rejected when scoring.

    This method searches for context IDs within the provided embeddings based on the given question.
    It then performs a rescore with the language model.
    If any invented (hallucinated) IDs are found, they are logged.
    Finally, the method returns the accepted and rejected IDs as a tuple or a (None, None) pair
    """
    span = trace.span(
        name="embedding-search",
        metadata={"database": "corpus"},
        input={"query": question},
    )

    ids_scores = corpus.embeddings.search(question, limit=CONTEXT_CHOICES)
    span.end(output=ids_scores)

    for row_id, score in ids_scores:
        logging.debug(score, corpus.data[row_id])

    while True:
        rescoring_context = [corpus.data[row_id] for row_id, _ in ids_scores]
        rescoring_prompt = context_rescoring_prompt(question, rescoring_context)
        prompt_length = len(llm.encoding.encode(rescoring_prompt))
        logging.debug(rescoring_prompt)
        if prompt_length <= MODEL_CONTEXT_LENGTH:
            break
        ids_scores = ids_scores[: len(ids_scores) // 2]

    try:
        # creates generation
        generation = trace.generation(
            name="context-rescoring",
            model=MODEL,
            #            model_parameters={"maxTokens": "1000", "temperature": "0.9"},
        )

        completion = llm.answer(
            rescoring_prompt,
            output_json=True,
            name="de-wiki-context",
            metadata={"question": question, "rescoring_context": rescoring_context},
        )

        generation.end(
            output=completion,
        )
    except openai.BadRequestError as e:
        logging.error("API wasn't happy: %s", e)
    else:
        try:
            try:
                _ = json.loads(completion)
                accepted_id_string = completion
            except json.JSONDecodeError:
                # While ChatGPT mostly correctly returns only the ids in JSON format,
                # some other models may add text before and after the chunk id list.
                accepted_id_string = next(
                    s
                    for s in completion.split("\n")
                    if s
                    and all(
                        all(ch.isdigit() or ch in "[]," for ch in sub)
                        for sub in s.split()
                    )
                )

                if "], [" in accepted_id_string:
                    # Another output format bug with Claude
                    accepted_id_string = accepted_id_string.replace("], [", ", ")

            try:
                returned_ids = json.loads(accepted_id_string)
                while isinstance(returned_ids, dict):
                    returned_ids = list(returned_ids.values())[0]
                assert isinstance(returned_ids, list) and all(
                    isinstance(i, int) for i in returned_ids
                )
            except (AssertionError, json.JSONDecodeError):
                returned_ids = [int(s) for s in accepted_id_string.split()]

            assert isinstance(returned_ids, list) and all(
                isinstance(i, int) for i in returned_ids
            )

            if invented_ids := set(returned_ids) - {row_id for row_id, _ in ids_scores}:
                logging.info(
                    f"The model invented following context IDs: {invented_ids}"
                )

            accepted_ids = [
                row_id for row_id in returned_ids if row_id not in invented_ids
            ]

            rejected_ids = set(cid for cid, _ in ids_scores) - set(accepted_ids)

            return accepted_ids, rejected_ids

        except (ValueError, AssertionError, StopIteration):
            logging.warning(
                "Received a response to '%s' that I cannot parse: '%s'",
                rescoring_prompt,
                completion,
            )

    return [], []


def run_loop(llm: LLM, corpus: Corpus, question: str):
    """Run an interactive loop to test the context retrieval"""

    langfuse = Langfuse()
    langfuse.auth_check()
    session_id = datetime.now().strftime("%Y%m%d-%H%M")

    while question:
        trace = langfuse.trace(
            name="de-wiki-context",
            input={"question": question},
            session_id=session_id,
        )
        logging.info("Answering '%s'", question)
        logging.info("Monitor trace in Langfuse: %s", trace.get_trace_url())

        context_ids, rejected_ids = get_context_ids(question, corpus, llm, trace)

        if context_ids:
            print("---- Accepted ----")
            for cid in context_ids:
                print(corpus.format_chunk(cid))

            print("---- Rejected ----")
            for cid in rejected_ids:
                print(corpus.format_chunk(cid))

            context = build_context(corpus.data[cid] for cid in context_ids)

            print("---- Without context ----")
            print(llm.answer(question_prompt(question), name="context-off"))

            print("---- With context ----")
            print(llm.answer(question_prompt(question, context), name="context-on"))

        question = input("---- Question: ")

    langfuse.flush()


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    dotenv.load_dotenv()

    if pulze_key := os.environ.get("PULZE_KEY"):
        client = OpenAI(api_key=pulze_key, base_url="https://api.pulze.ai/v1")
    else:
        client = OpenAI()

    initial_question = random.choice(INITIAL_QUESTIONS)
    run_loop(LLM(client, MODEL, MAX_ANSWER_TOKENS), load_corpus(), initial_question)
