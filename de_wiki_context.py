#!/usr/bin/env python
"""Generate context using German Wikipedia articles.

We take top 10% paragraphs (ranked by article views), embed them using a large 
multilingual model, and cache the embeddings. For each question we retrieve the pieces 
of context and rescore them using Pulze API. We demonstrate what the answers look with 
and without using the context.
 """

import logging
import os
import random

import dotenv
import openai
import tiktoken
from datasets import load_dataset, load_from_disk
from openai import OpenAI
from txtai import Embeddings

INITIAL_QUESTIONS = [
    "How many wives can a man have in Germany?",
    "What are the parties in current German parliament?",
    "Who is in the current German government?",
    "Wer ist ein Schöffe bzw eine Schöffin?",
    "Was waren die deutsch-französischen Beziehungen im 19. Jhd?",
    "Why was the Berlin wall built?",
]

DATASET_SOURCE = "Cohere/wikipedia-22-12"
DATASET_PATH = "data/de-wiki-22-12-cohere-by-views"

# much better than the default one for German text
EMBEDDINGS_MODEL = "intfloat/multilingual-e5-large"
EMBEDDINGS_HOW_MANY_K = 1500  # total size of the dataset is 15M embeddings
EMBEDDINGS_PATH = f"data/de-wiki-multilingual-e5-large-top-{EMBEDDINGS_HOW_MANY_K}k"

CONTEXT_CHOICES = 20
OPENAI_MODEL = "pulze"
OPENAI_MODEL_CONTEXT_LENGTH = 8191


def load_data_embeddings():
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

    return data, embeddings


def build_context(context_chunks):
    """Prepare a context string out of the suggested content chunks"""
    return "\n".join(
        f"""{c["id"]} (from '{c["title"]}'): {c["text"]}""" for c in context_chunks
    )


def context_rescoring_prompt(query, context_chunks):
    """Prepare a rescoring prompt for context chunks"""
    return f"""
You are part of a text retrieval engine for German language. Your goal is to check whether
the context, retrieved from the vector database, is helpful when answering the 
query asked.

The query: {query}

Context pieces, taken from Wikipedia articles, that you need to check:
 {build_context(context_chunks)}
 
Provide the list of ids of context pieces that help answer the question posed, 
separated by space. Do not give any other output. Example: 7682345 23876423 324123
   """


def question_prompt(query, context_string=None):
    """Prepare a question prompt that optionally includes a context"""
    return f"""
You are a question-answer engine who takes great care to provide the most accurate answer. 
Answer the following question in German to the best of your ability: {query}
Aim at several paragraphs that show clear and reasoned thinking. 
    """ + (
        ""
        if not context_string
        else """
The following context pieces, taken from recent Wikipedia articles, might be helpful in the answer:
"""
        + context_string
    )


def run_loop(client, data, embeddings, question):
    """Run an interactive loop to test the context retrieval"""
    try:
        encoding = tiktoken.encoding_for_model(OPENAI_MODEL)
    except KeyError:
        encoding = tiktoken.encoding_for_model("gpt-4")

    def complete(prompt):
        return (
            client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=OPENAI_MODEL,
            )
            .choices[0]
            .message.content
        )

    def format_chunck(chunk_id):
        return f"""{chunk_id} [{data[chunk_id]["title"]}] {data[chunk_id]["text"]}"""

    while question:
        logging.info("Answering %s", question)

        ids_scores = embeddings.search(question, limit=CONTEXT_CHOICES)
        for row_id, score in ids_scores:
            logging.debug(score, data[row_id])

        while True:
            rescoring_prompt = context_rescoring_prompt(
                question, (data[row_id] for row_id, _ in ids_scores)
            )
            prompt_length = len(encoding.encode(rescoring_prompt))
            logging.debug(rescoring_prompt)
            if prompt_length <= OPENAI_MODEL_CONTEXT_LENGTH:
                break
            ids_scores = ids_scores[: len(ids_scores) // 2]

        try:
            completion = complete(rescoring_prompt)
        except openai.BadRequestError as e:
            logging.error("API wasn't happy: %s", e)
        else:
            try:
                print("---- Accepted ----")
                accepted_ids = [int(s) for s in completion.split()]
                for cid in accepted_ids:
                    print(format_chunck(cid))

                print("---- Rejected ----")
                rejected_ids = set(cid for cid, _ in ids_scores) - set(accepted_ids)
                for cid in rejected_ids:
                    print(format_chunck(cid))

                context = build_context(data[cid] for cid in accepted_ids)

                print("---- Without context ----")
                print(complete(question_prompt(question)))

                print("---- With context ----")
                print(complete(question_prompt(question, context)))

            except ValueError:
                logging.warning(
                    "Received a response that I cannot parse: %s", completion
                )

        question = input("Question: ")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

    env = dotenv.dotenv_values()
    client_ = OpenAI(api_key=env["PULZE_API_KEY"], base_url="https://api.pulze.ai/v1")
    data_, embeddings_ = load_data_embeddings()

    initial_question = random.choice(INITIAL_QUESTIONS)
    run_loop(client_, data_, embeddings_, initial_question)
