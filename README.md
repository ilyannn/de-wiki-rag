# Context retrieval example

### [de_wiki_context.py](de_wiki_context.py)

- Pulze key is required in the `.env` file
- Running it for the first time will download the de-wiki dataset and compute the embeddings
- You can set `EMBEDDINGS_HOW_MANY_K` to a lower value if you don't want to wait
- Embeddings are cached in the `data/` folder
- A few sample questions are provided
