GPT Confluence Challenge
========================

### Installation
1. Create new python environment
2. Install all dependencies: `pip install -r requirements.txt`
3. Download the model: `mkdir -p models && wget -O models/ggml-nous-gpt4-vicuna-13b.bin https://gpt4all.io/models/ggml-nous-gpt4-vicuna-13b.bin`

### Create confluence embeddings from static html
1. The pages are already stored in a folder called `html_confluence`
2. Create embeddings via `python ingest.py html_confluence`
3. Run `python main.py` and ask a question (e.g. "What is the difference between agile and scrum?"). Be aware that the might take a minute or more.
