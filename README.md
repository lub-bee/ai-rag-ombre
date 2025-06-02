# Memo

1. Start the DB server `docker compose up -d`
2. Start the LLM `ollama run phi3:mini`
3. Start the project `poetry run python3 main_lore.py`

## Test script

### Singlestoredb container, database and table
To test manually if the DB is reachable and exists, you can run : 
`poetry run python3 test.py`

## Other

`main.py` is the initital attemp, kept for testing purpose. Do not use, extremly heavy, embed & db reset at each question.