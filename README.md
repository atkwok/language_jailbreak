# language_jailbreak

This directory holds notebooks and scripts used to study and experiment with language model jailbreak prompts. The current plan is to gather a dataset of jailbreak prompts in English, translate them into other languages, run them against a set of open-source models, translate the outputs back to English, and measure attack success rates across languages and models.

New collaborators:
- Create an account at https://colab.research.google.com/ so you can run the notebooks in Google Colab.
- Request access to this repository from the maintainers, then clone or fork it to your GitHub account.
- Open the notebook from your GitHub copy and run it directly in Colab once access is approved.

Structure so far:
- `colab_llm.ipynb`: Colab-friendly notebook for jailbreak experiments.
- `data/models.json`: Dummy list of model targets to evaluate.
- `data/prompts.en.json`: Dummy list of English jailbreak prompts to seed translations.
- `translate.py`: Helper to translate prompts into other languages and save them for downstream testing.
- `translations/`: Placeholder directory to store translated prompt files produced by `translate.py`.
