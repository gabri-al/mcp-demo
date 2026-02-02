# Demo - Creating a custom MCP server on Databricks, hosted on Databricks Apps
This repo includes a demo of a custom MCP-server and related agent, hosted on Databricks.

The agent uses tools from the [financial dataset mcp](https://www.financialdatasets.ai/), therefore, a valid API key is required.

## Folder structure
* The `mcp-server` folder contains the code relevant to deploy the server as a Databricks Apps. This was created starting from [this repo](https://github.com/databrickslabs/mcp/tree/master/examples/custom-server) and following requiremens and docs [here](https://docs.databricks.com/aws/en/generative-ai/mcp/custom-mcp).
The App should be created as **custom app** and then **deployed providing this folder path**; the app's name should start with `mcp-`.

* The `tools-local-tests` folder contains the mcp tools that have been tested locally as python functions.

* The `mcp-agent-endpoint` folder contains the code needed to create an agent (as UC MLflow model) and then deploy it as an endpoint. It is based on LangGraph and the code comes from the notebook attached to [this documentation page](https://docs.databricks.com/aws/en/generative-ai/mcp/custom-mcp). There are some preliminary steps required here:
  1. Create a Service principal and add secret to it: Workspace settings > Identity and access > Service principals > Add service principal
  2. Grant Access on the MCP Server (Databricks Apps) to the Service Principal: open the permission settings of your Databricks Apps and add your service principal (look for the Client ID generate at the step above)
  3. Create / Update a secret scope with Service Principal Info (this is done in Notebook 00_ within this folder, using a .env file there)
  4. Review the langgraph notebook to adjust all the TODO; note that cell 4 is creating an agent.py file locally using the variables defined in cell 9

## Demo script:

### Playground --> MCP
- From Playground test any model with questions like: "What is today's Nvidia stock price?"
- Then, assign the MCP Server and ask: "Tell me what tools you have available now?"
- Next, compare another model without MCP server access and ask: "What is today's Nvidia stock price?"
- Additional question: "Tell me the average stock price of Nvidia in the last week?"
- Now add a UC function (this is managed MCP server) to evaluate trends and ask:
  - What is the performance of my Nvidia stocks, considering I purchased them on 15 April 2020?
  - What is the performance of my Apple stocks, considering I purchased them on 03 March 2020?
  - What is the performance of my Microsoft stocks, considering I purchased them on 15 January 2020?

### Model Serving Endpoint