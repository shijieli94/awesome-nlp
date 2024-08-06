import os

from awesome_nlp.constants import init_openai_client

API_KEY = init_openai_client(demo=True, return_key=True)

if os.getenv("OPENAI_API_KEY") is None:
    os.environ["OPENAI_API_KEY"] = API_KEY
assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "This doesn't look like a valid OpenAI API key"
print("OpenAI API key configured")

from wandb.integration.langchain import WandbTracer

wandb_config = {"project": "wandb_prompts_quickstart"}

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI

llm = OpenAI(
    temperature=0,
    openai_api_key=API_KEY,
    openai_api_base="https://api.chatanywhere.tech/v1",
    model_name="gpt-3.5-turbo",
)

tools = load_tools(["llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

questions = [
    "Find the square root of 5.4.",
    "What is 3 divided by 7.34 raised to the power of pi?",
    "What is the sin of 0.47 radians, divided by the cube root of 27?",
    "what is 1 divided by zero",
]
for question in questions:
    try:
        answer = agent.run(question, callbacks=[WandbTracer(wandb_config)])
        print(answer)
    except Exception as e:
        print(e)
        pass

WandbTracer.finish()

import wandb
from wandb.sdk.data_types import trace_tree

parent_span = trace_tree.Span(name="Example Span", span_kind=trace_tree.SpanKind.AGEN)

# Create a span for a call to a Tool
tool_span = trace_tree.Span(name="Tool 1", span_kind=trace_tree.SpanKind.TOOL)

# Create a span for a call to a LLM Chain
chain_span = trace_tree.Span(name="LLM CHAIN 1", span_kind=trace_tree.SpanKind.CHAIN)

# Create a span for a call to a LLM that is called by the LLM Chain
llm_span = trace_tree.Span(name="LLM 1", span_kind=trace_tree.SpanKind.LLM)
chain_span.add_child_span(llm_span)
tool_span.add_named_result({"input": "search: google founded in year"}, {"response": "1998"})
chain_span.add_named_result({"input": "calculate: 2023 - 1998"}, {"response": "25"})
llm_span.add_named_result(
    {
        "input": "calculate: 2023 - 1998",
        "system": "you are a helpful assistant",
    },
    {"response": "25", "tokens_used": 218},
)

parent_span.add_child_span(tool_span)
parent_span.add_child_span(chain_span)

parent_span.add_named_result({"user": "calculate: 2023 - 1998"}, {"response": "25 years old"})
run = wandb.init(name="manual_span_demo", project="wandb_prompts_demo")
run.log({"trace": trace_tree.WBTraceTree(parent_span)})
run.finish()
