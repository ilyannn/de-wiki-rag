# Context retrieval example

## [de_wiki_context.py](de_wiki_context.py)

- Pulze key is required in the `.env` file for access to the `pulze` model
- Langfuse key is required in the `.env` file for observability
- The first time will download the dataset and compute the embeddings
- You can set `EMBEDDINGS_HOW_MANY_K` to a lower value if you don't want to wait
- Embeddings are cached in the `data/` folder
- A few sample questions are provided

## [gen_context.py](gen_context.py)

This will process all the questions from the main project.
