from zep_cloud import Message, EntityEdge, EntityNode, Episode
from functools import wraps
from openai import AzureOpenAI
from openai import OpenAI
from zep_cloud.types import Message
import logging

logger = logging.getLogger(__name__)

TEMPLATE = """
FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts and their valid date ranges. If the fact is about an event, the event takes place during this time.
# format: FACT (Date range: from - to)

{facts}


# These are the most relevant entities
# ENTITY_NAME: entity summary

{entities}


# These are the most relevant episodes.
# format: EPISODE

{episodes}

"""


def format_edge_date_range(edge: EntityEdge) -> str:
    # return f"{datetime(edge.valid_at).strftime('%Y-%m-%d %H:%M:%S') if edge.valid_at else 'date unknown'} - {(edge.invalid_at.strftime('%Y-%m-%d %H:%M:%S') if edge.invalid_at else 'present')}"
    return f"{edge.valid_at if edge.valid_at else 'date unknown'} - {(edge.invalid_at if edge.invalid_at else 'present')}"


def compose_search_context(edges: list[EntityEdge] | None, nodes: list[EntityNode] | None, context_block: str, episodes: list[Episode] | None) -> str:
    edges = edges or []
    nodes = nodes or []
    episodes = episodes or []
    facts = [f'  - {edge.fact} ({format_edge_date_range(edge)})' for edge in edges if edge]
    entities = [f'  - {node.name}: {node.summary}' for node in nodes if node]
    episodes = [f'  - Content: {episode.content}' for episode in episodes if episode]
    return TEMPLATE.format(facts='\n'.join(facts), entities='\n'.join(entities), context_block=context_block, episodes='\n'.join(episodes))



def except_retry_dec(retry_num: int = 3):
    def decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            i = 0
            while True:
                try:
                    logger.info("openai agent post...")
                    ret = func(*args, **kwargs)
                    logger.info("openai agent post finished")
                    return ret
                # error define: https://platform.openai.com/docs/guides/error-codes/python-library-error-types
                except (
                    openai.BadRequestError,
                    openai.AuthenticationError,
                ) as e:
                    raise
                except Exception as e:  # pylint: disable=W0703
                    logger.error(f"{e}")
                    logger.info(f"sleep {i + 1}")
                    time.sleep(i + 1)
                    if i >= retry_num:
                        raise
                    logger.warning(f"do retry, time: {i}")
                    i += 1

        return wrapped_func

    return decorator


class OpenAIAgent:
    def __init__(
        self, model, source, api_dict, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

        if source == "azure":
            self.client = AzureOpenAI(
                azure_endpoint = api_dict["endpoint"], 
                api_version=api_dict["api_version"],
                api_key=api_dict["api_key"],
                )
        elif source == "openai":
            self.client = OpenAI()
        elif source == "deepseek":
            self.client = OpenAI(
                    # This is the default and can be omitted
                    base_url=api_dict["base_url"],
                    api_key=api_dict["api_key"],
                )
        print(f"You are using {self.model} from {source}")
        
    @except_retry_dec()
    def generate(self, messages: list[dict], max_new_tokens:int=None) -> str:
        _completion = self.client.chat.completions.create(
                messages=messages,
                temperature=self.temperature,
                model=self.model,
            )
        return [_completion.choices[0].message.content]




async def llm_response(llm_client, context: str, question: str) -> str:
    system_prompt = """
        You are a helpful expert assistant answering questions from users based on the provided context.
        """

    prompt = f"""
            Your task is to briefly answer the question. You are given the following context from the previous conversation. If you don't know how to answer the question, abstain from answering.
                
                {context}
                
                
                {question}
                

            Answer:
            """

    # llm_client.generate is sync, so run it in a thread to avoid blocking the event loop
    import asyncio
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: llm_client.generate(
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": prompt}]
        )
    )
    result = response[0] or ''

    return result

# since zep has a 400 chars limit
def get_retrieval_query(query: str) -> str:
    import re
    # Prefer extracting the segment starting from the LAST occurrence of
    # "These are the events" up to (but not including) the EARLIEST of
    # the two end markers below. This avoids capturing earlier duplicated sections
    # and excludes any trailing task/list sections.
    start_marker = "These are the events"
    end_markers = [
        "Your task is to",
        "Below is a list of possible subsequent events:",
    ]
    end_indices = [idx for m in end_markers if (idx := query.find(m)) != -1]
    if end_indices:
        end_idx = min(end_indices)
        start_idx = query.rfind(start_marker, 0, end_idx)
        if start_idx != -1:
            return query[start_idx:end_idx].strip()
    
    # Others
    match = re.search(r"Now Answer the Question:\s*(.*)", query, re.DOTALL)
    if match:
        retrieval_query =  ''.join(match.groups())
    else:
        match = re.search(r"Here is the conversation:\s*(.*)", query, re.DOTALL)
        if match:
            retrieval_query =  ''.join(match.groups())
        else:
            retrieval_query = query
            
    return retrieval_query


def construct_messages(content: str, user_id: str) -> list[Message]:
    messages = [
                Message(
                    name=f"{user_id}",
                    content=content[:2400],  # content is limited to 2400 characters / unpaid user
                    role="user",
                ),
                Message(
                    name="AI Assistant",
                    content="Hello! I will memorize the content for you.",
                    role="assistant",
                )
                    ]
    return messages