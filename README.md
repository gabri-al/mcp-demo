# Creating MCP on Databricks
This repo includes the files needed to run a demo of a custom MCP-server on Databricks.

## Folder structure
* The `mcp-server` folder contains the code relevant to deploy the server as a Databricks Apps. This was created starting from [this repo](https://github.com/databrickslabs/mcp/tree/master/examples/custom-server). The App should be deployed as custom app and then deploying it providing this folder path.

* No API keys are actually required (the local env is just a placeholder)

* The `tools-local-tests` folder contains the mcp tools that have been tested locally as python functions.

* Demo scripts:
- From Playground test any model with questions like: "What is today's Nvidia stock price?"
- Then, assign the MCP Server and ask: "Tell me what tools you have available now?"
- Next, compare another model without MCP server access and ask: "What is today's Nvidia stock price?"
- Additional question: "Tell me the average stock price of Nvidia in the last week?"
- Now add a UC function (this is managed MCP server) to evaluate trends and ask:
  - What is the performance of my Nvidia stocks, considering I purchased them on 15 April 2020?
  - What is the performance of my Apple stocks, considering I purchased them on 03 March 2020?
  - What is the performance of my Microsoft stocks, considering I purchased them on 15 January 2020?