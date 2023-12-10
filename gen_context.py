import logging
import os
import sys

from openai import OpenAI
from dotenv import load_dotenv
from redis import Redis

# Local imports
from de_wiki_context import get_context_ids, load_corpus
from llm import LLM

# Imports from the main project
load_dotenv()
sys.path.append(os.environ["MAIN_PROJECT"])
from access_redis import (
    create_redis_client,
    get_redis_context_ids,
    put_redis_context_ids,
)
from data.read_data import questions


MAX_QUESTIONS = 25


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

    api_key = os.environ["PULZE_API_KEY"]
    client = OpenAI(api_key=api_key, base_url="https://api.pulze.ai/v1")

    corpus = load_corpus()
    llm = LLM(client, "pulze", 1000)

    redis_: Redis = create_redis_client()

    existed = 0
    saved = 0
    saved_chunks = 0
    errored = 0

    try:
        for _, question in zip(range(MAX_QUESTIONS), questions()):
            print()
            print(question)
            chunk_ids = get_redis_context_ids(redis_, question.question_id)

            if chunk_ids:
                print("Already there, skipping getting context")
                existed += 1
                continue

            chunk_ids, _ = get_context_ids(question.phrase, corpus, llm)

            if not chunk_ids:
                print("Did not get context")
                errored += 1
                continue

            chunk_texts = [corpus.data[cid]["text"] for cid in chunk_ids]
            put_redis_context_ids(redis_, question.question_id, chunk_ids, chunk_texts)

            for cid in chunk_ids:
                print(corpus.format_chunk(cid))

            saved += 1
            saved_chunks += len(chunk_ids)
    finally:
        print(f"Saved {saved_chunks} context chunks for {saved} questions")
        print(f"There existed {existed} questions with context already provided")
        print(f"Weren't able to retrieve context for {errored} questions")
