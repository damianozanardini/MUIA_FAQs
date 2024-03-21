Source: https://github.com/NVIDIA/GenerativeAIExamples/tree/main/examples



Create a python virtual environment and activate it

> python3 -m venv genai
> source genai/bin/activate

Goto the root of this repository GenerativeAIExamples and execute below command to install the requirements

> pip install -r requirements.txt

API Key for using the mixtral_8x7b LLM

> export NVIDIA_API_KEY="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"

Run the example using streamlit

> streamlit run main.py