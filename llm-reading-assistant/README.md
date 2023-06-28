# LLM Raeding Assistant

## Overview

LLM Reading Assistant is a simple demo of an app which purpose is to facilitate reading complex documents from potentially various domains. Such documents usually contain a lot of domain specific terms and wording as well as many references to other documents or to specialized domain knowledge. Examples of such documents might be:  

- legal acts
- scientific papers
- medical documentation
- financial reports

 Reading these documents, especially by a person that is not a top expert in the field, usually means a lot of time spent on searching for auxiliary information and definitions in outside sources. The capabilities of Large Language Models, that are pretrained on vast corpora of texts from different areas and encode this knowledge in their weights, also allowing for easy extraction of information by interacting with them in natural language, make them a solid backbone for building a tool that can make understanding domain specific texts a lot faster and easier.

## How it works

The idea presented in this demo is as simple as follows:
- You upload the document that you want to read into the web-based app and start reading
- When you encounter any incomprehensible term or hard to understand portion of text, you can select it and ask LLM to either explain or summarize it
- An appropriate prompt will be constructed under the hood, sent via an API and the answer will be returned and printed

![Reading Assistant Streamlit GUI](/img/reading_assistant_gui.png)

The following demo is just a PoC showing the idea and presenting a specific mix of technologies (see the next chapter) that can be used to build similar solutions. It was build according to [QuickStart ML](https://github.com/getindata/quickstart-ml-blueprints#overview) principles, with just one exception of using paid APIs to commercial black-box Large Language Models from top providers (OpenAI, Google). To be able to reforge it into a production grade solution, a few additional developments would be needed:
- a more advanced user interface allowing for better user experience using context menus instead of manual copy-paste operations
- possibly a chat window to be able to extend communication with the model beyond simple explain/summarize queries
- an option to use large-context models and in-document search in addition to just relying on pretrained model knowledge
- comprehensive load and functional tests
- optionally the use of open-source, self-deployed models, finetuned on domain specific corpora

## Technologies

This use case depends on the following technologies:
- the usual [QuickStart ML](https://github.com/getindata/quickstart-ml-blueprints#overview) tech stack including [Kedro](https://kedro.org/) for pipelining, configuration management etc.
- [Streamlit](https://streamlit.io/) to easily build a user interface layer that also nicely works together with Kedro-based backend
- APIs to commercial Large Language Models, currently including:
    - [OpenAI API](https://openai.com/blog/openai-api) - direct access to the family of GPT models from OpenAI
    - [Azure OpenAI API](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/) - access to the same models, but using Microsoft Azure services that might be more suitable for some organizations due to security reasons and cost control
    - [VertexAI PaLM API](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/text-bison?project=gid-ml-framework) - an alternative to OpenAI from Google, accessed via Google Cloud Platform's VertexAI service and enabling usage of PaLM models (there is also a possibility of using direct API service, however since it is in preview phase, it is currently limited to the US and there is a waitlist so this option is no implemented yet in the demo).

### Disclaimers

#### Issues with Streamlit in containerized setup on Apple M1/M2

As with other solutions based on QuickStart ML Blueprints, it is possible to run and further develop this example inside [VSCode's Dev Container](https://code.visualstudio.com/docs/devcontainers/containers). There are also [usual issues](https://github.com/getindata/quickstart-ml-blueprints#howtostart-local-remarks) related to that way of working in specific setups. Additionally, one extra issue was observed regarding Dev Containers on Apple M1/M2 processors [which is related to Streamlit](https://github.com/streamlit/streamlit/issues/4842). Although in the uncontainerized setup (like [here](https://github.com/getindata/quickstart-ml-blueprints#howtostart-local-alt)) everything works well about the Streamlit app, when containerized using Docker and Dev Containers it might throw an error:

```
OSError: [Errno 38] Function not implemented
```

Dev Containers were tested in other standard local setups (Linux, Windows, Intel Mac) and no additional issues were found, especially related to using Streamlit framework.

#### Issues with browsers

Streamlit application provides a feature of displaying an uploaded PDF file for reading. This feature was identified to cause problems on [Chrome browser](https://discuss.streamlit.io/t/pdf-reader-problems/36081/14). The PDF simply sometimes won't show properly and the user will see an empty space instead. No issues were found when testing on the following browsers: Firefox, Safari and Microsoft Edge.

## Pipelines and configuration

The entire solution consists of just a single Kedro pipeline (`run_assistant`) with a single node (`complete_request`) inside. The usual way of running Kedro pipelines is to use `kedro run`, but in this case there is another element on top of Kedro - a Streamlit application. In practice, in `src` directory there is an additional script `run_app.py` that:

- handles additional parameters that are not passed to the Kedro pipeline via Kedro `conf` file, but via the Streamlit app instead
- defines web-UI appearance and widgets
- runs Kedro pipeline

So instead of calling `kedro run`, the user needs to call `streamlit run` command to run the app, and then use GUI to interact with the application by changing parameters and executing Kedro pipeline underneath (the details on how to run the app are given in the next chapter).

`complete_request` node that is contained by `run_asssistant` pipeline has 6 parameters, 2 of which are provided in traditional way via Kedro configuration in `conf/base/parameters/run_assistant.yml` file and the remaining 4 are passed via Streamlit interface. This also means, that the usual `kedro run` won't work by default, because 4 Streamlit parameters will be missing (unless they are provided using `--params` argument, but it is not the intended way). The parameters are (in square brackets there is a way of injecting them to the pipeline):

- `api` [Streamlit app]: API to the LLM model to be used; Currently supported APIs are: `OpenAI`, `VertexAI PaLM`, `Azure OpenAI`
- `mode` [Streamlit app]: action to be performed with the input text; Currently either `explain` or `summarize`
- `input_text` [Streamlit app]: the text from the document that the user wants to explain or summarize
- `instructions` [Kedro config]: additional instructions for the LLM model that define and precise explanation or summarization task. Depending on the selected `mode`, a proper instruction is appended before the `input_text` to create a final prompt for the model
- `max_tokens` [Kedro config]: the limit of tokens to be generated by the model; it can be different for different tasks
- `model` [Streamlit app]: the name of the model to be used behind the selected API

Depending on the value of the `api` parameter, `complete_request` node (in order to send API request and retrieve the response) uses an object of a proper class from among: `OpenAIAPIRequest`, `VertexAIPaLMAPIRequest` and `AzureOpenAIAPIRequest` that are defined in `src/llm_reading_assistant/pipelines/run_assistant/requests.py`.

To use Azure OpenAI Service API one additional parameter needs to be provided in `conf/base/parameters.yml` which is the list of Azure OpenAI deployments (details in the next chapter) under the name `azure_openai_deployments`.

Also, to use OpenAI services (both native or in Azure) some credentials are needed. Credentials are stored in `conf/local/credentials.yml` file which is excluded from Git storage. The credentials are:
- for native OpenAI API: `openai_api_key`
- for Azure OpenAI Service: `azure_openai_api_key` and `azure_openai_api_endpoint`

## How to run

Currently there are 3 APIs that can be used to interact with different LLMs:  
- [OpenAI API](https://openai.com/blog/openai-api)
- [Azure OpenAI Service API](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/)
- [VertexAI PaLM API](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/text-bison?project=gid-ml-framework)  
   

1. **OpenAI API** 

To be able to use app via OpenAI an API key is needed. It should be placed in `conf/local/credentials.yml` file under the name: `openai_api_key`. This file is not stored in Git. OpenAI API key can be found or created here after you create and configure  OpenAI API account: https://platform.openai.com/account/api-keys.

2. **Azure OpenAI Service API**

To use Azure OpenAI API you need to have access (or create it by yourself) an [Azure OpenAI Resource](https://portal.azure.com/#view/Microsoft_Azure_ProjectOxford/CognitiveServicesHub/~/OpenAI).

Then, you need to go to "Model Deployments" → "Manage Deployments" and check which models (and under what names) are deployed within your Azure OpenAI Resource. If none, you need to create one.

After you confirm having at least one deployed model, you need to copy its name (or all names that you want to later choose from) and put it as a list in `conf/base/parameters.yml` file under the name `azure_openai_deployments`.

The last step is to click on your Azure OpenAI Resource and then "Click here to manage keys". You need to copy `KEY 1` and `Endpoint` fields and paste them in `conf/local/credentials.yml` file under the names `azure_openai_api_key` and `azure_openai_api_endpoint`, respectively.

3. **VertexAI PaLM API**

To use Google PaLM 2 model via VertexAI, you need to have a Google Cloud account and a project. You need to [set up authentication for a local development](https://cloud.google.com/docs/authentication/provide-credentials-adc?&_ga=2.236834051.-348040172.1646755094#local-dev) using [`gcloud` utility](https://cloud.google.com/sdk/docs/install), which basically means running `gcloud auth application-default login` command from your local console and following instructions. No other credentials need to be provided.

---

After setting up the access to selected API, you need to [recreate the working environment](https://github.com/getindata/quickstart-ml-blueprints#running-existing-project-locally-) from `poetry.lock` or `pyproject.toml` file according to Quickstart ML way of work:

- preferably [using Dev Containers](https://github.com/getindata/quickstart-ml-blueprints#recommended-way-using-vscode-and-dev-containers-)
- or [manually using Poetry](https://github.com/getindata/quickstart-ml-blueprints#alternative-ways-of-manual-environment-creation-)

When the working environment it set up  run Streamlit app and use the GUI in your browser:

```bash
streamlit run src/run_app.py
```
