# Pre-requisites
- Create or reuse a storage account
- Import the demo data in the storage account
  - default container name: `demo`
  - cloud files:
    - job metadata
    - category
    - annotated & unannotated data
    - metrics
  - all above files will have a copy in `download` folder when you run the API
- Put connection string of the storage account into main.py as `default_connection_string`or set it as an environment variable like below:
  - Shell: `export CONNECTION_STRING=<connection_string>`
  - Python: `os.environ['CONNECTION_STRING'] = <connection_string>`


# Run the API
- You can run the API with docker containers or locally.
- If you could test the API locally, you can execute:
  - `uvicorn main:app --reload`
- It will be running on `http://127.0.0.1:8000` or other available port.
- APIs can be accessed and tested via `http://127.0.0.1:8080/docs#/` or other available port.
  - ![FastAPI Docs Sample](../../gpt_ale/docs/images/gpt_ale_api_docs.png)