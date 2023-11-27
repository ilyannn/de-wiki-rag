#!/usr/bin/env python

import logging
import os

import openai
import tiktoken
from dotenv import load_dotenv

from datasets import load_dataset, load_from_disk
from txtai import Embeddings
from openai import OpenAI

DATASET_PATH = "data/de-wiki-22-12-cohere-by-views"

EMBEDDINGS_MODEL = (
    "intfloat/multilingual-e5-large"  # much better than the default one for German text
)
EMBEDDINGS_HOW_MANY_THOUSANDS = 1500

EMBEDDINGS_PATH = (
    f"data/de-wiki-multilingual-e5-large-top-{EMBEDDINGS_HOW_MANY_THOUSANDS}k"
)

CONTEXT_CHOICES = 20
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_MODEL_CONTEXT_LENGTH = 4096


def load_data_embeddings():
    try:
        data = load_from_disk(DATASET_PATH, keep_in_memory=True)
        logging.info(f"Loaded data of shape {data.shape} from {DATASET_PATH}")
    except FileNotFoundError:
        full_data = load_dataset("Cohere/wikipedia-22-12", "de", split="train")
        logging.info(f"Loaded full wikipedia data of shape {full_data.shape}")
        data = full_data.sort(["views"], reverse=True, kind="stable")
        os.makedirs(DATASET_PATH, exist_ok=True)
        data.save_to_disk(DATASET_PATH)
        logging.info(f"Saved data of shape {data.shape} to {DATASET_PATH}")

    embeddings = Embeddings(path=EMBEDDINGS_MODEL)
    logging.info(f"Loaded embedding model {EMBEDDINGS_MODEL}")

    try:
        embeddings.load(EMBEDDINGS_PATH)
        logging.info(f"Loaded {embeddings.count()} embeddings from {EMBEDDINGS_PATH}")
    except FileNotFoundError:
        top_data = next(data.iter(EMBEDDINGS_HOW_MANY_THOUSANDS * 1000))
        embeddings.index(top_data["text"])
        embeddings.save(EMBEDDINGS_PATH)
        logging.info(f"Saved {embeddings.count()} embeddings to {EMBEDDINGS_PATH}")

    return data, embeddings


def context_rescoring_prompt(query, context_chunks):
    def chunk_prompt(c):
        return f"""
{c["id"]} (from '{c["title"]}'): {c["text"]}"""

    return f"""
You are part of a text retrieval engine. Your goal is to check whether
the context, retrieved from the vector database, is helpful when answering the 
query asked. The query as well as context is in German. 

The query: {query}

Context pieces that you need to check:
 {''.join(chunk_prompt(c) for c in context_chunks)}
 
Provide the list of ids of context pieces that help answer the question posed, separated by space.
   """


def run_loop(data, embeddings, question):
    client = OpenAI()
    encoding = tiktoken.encoding_for_model(OPENAI_MODEL)

    def format_chunck(cid):
        return f"""{cid} [{data[cid]["title"]}] {data[cid]["text"]}"""

    while question:
        logging.info("Answering %s", question)

        ids_scores = embeddings.search(question, limit=CONTEXT_CHOICES)
        for row_id, score in ids_scores:
            logging.debug(score, data[row_id])

        while True:
            prompt = context_rescoring_prompt(
                question, (data[row_id] for row_id, _ in ids_scores)
            )
            prompt_length = len(encoding.encode(prompt))
            logging.debug(prompt)
            if prompt_length <= OPENAI_MODEL_CONTEXT_LENGTH:
                break
            ids_scores = ids_scores[: len(ids_scores) // 2]

        try:
            completion = (
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

            print("---- Accepted: ----")
            accepted_ids = [int(s) for s in completion.split()]
            for cid in accepted_ids:
                print(format_chunck(cid))

            print("---- Rejected: ----")
            rejected_ids = set(cid for cid, _ in ids_scores) - set(accepted_ids)
            for cid in rejected_ids:
                print(format_chunck(cid))

        except openai.BadRequestError as e:
            logging.error("API wasn't happy: %s", e)
        except ValueError:
            logging.warning("Received a response that I cannot parse: %s", completion)

        question = input("Question: ")


if __name__ == "__main__":
    #   openai.api_base = "https://api.pulze.ai/v1"
    load_dotenv()

    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    data_, embeddings_ = load_data_embeddings()
    run_loop(
        data_,
        embeddings_,
        question="Was waren die deutsch-französische Beziehungen im 19. Jhd?",
    )  # "Why was the Berlin wall built?")
