from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langchain_nebius import ChatNebius
from typing import List, Annotated
from operator import add
from pydantic import BaseModel, Field
from typing import List, Literal
from config import NEBIUS_API_KEY
from app import query_rag_system

SYSTEM_PROMPT = {"role":"system", "content":"You are a retrieval augemented generation chatbot. Your job is to sythesize information from DOCUMENTs to use as additional context for answering the user's QUESTION."}

INDEX_NAME = "rag-assistant-0"

class ConceptGuide(BaseModel):
    concept_name: str = Field(description="The name of the concept to be studied")
    concept_description: str = Field(description="A description of the concept to be studied")
    cited_pages: list[str] = Field(description="A list of concept relevant pages to be studied")


class StudyGuide(BaseModel):
    original_query: str = Field(description="The original query from the user")
    study_guide: list[str] = Field(
        description="A list of concepts to be studied by the user pertaining to the original query")


class ContextFeedback(BaseModel):
    grade: Literal["Pass", "Rewrite", "More Context"] = Field(
        description="Grade: Pass (good enough),Rewrite (has content but needs improvement),More Context (needs additional information)")
    feedback: str = Field(description="A few sentences of feedback on the study guide")


class State(TypedDict):
    grade: str = None
    feedback: Annotated[List[str], add]
    study_guide: StudyGuide
    original_query: str
    # this allows for procedural context generation
    retrieved_context: Annotated[List[str], add]
    # we should have new queries here as well
    new_query: Annotated[List[str], add]
    num_iterations: int = 0
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# tool llm
llm = ChatNebius(
    api_key=NEBIUS_API_KEY,
    model="Qwen/Qwen3-4B-fast",
    temperature=0.0,
 )
llm_with_tools = llm.bind_tools([query_rag_system], tool_choice="required")

# instance to generate study guide
study_guide_llm = ChatNebius(
    api_key=NEBIUS_API_KEY,
    model="Qwen/Qwen3-4B-fast",
    temperature=0.0,
 )
study_guide_gen = study_guide_llm.with_structured_output(StudyGuide)

# feedback llm
feedback_llm_instance = ChatNebius(
    api_key=NEBIUS_API_KEY,
    model="Qwen/Qwen3-4B-fast",
    temperature=0.0,
)
feedback_llm = feedback_llm_instance.with_structured_output(ContextFeedback)

# langchain node definitions
def query_generator_and_call(state: State, query_rag_system):
    """
    Handles writing queries and issuing tool all requests.
    if the user has started the chain, use most recent message as the prompt for the query
    if we've already tried to generate a query, and we need a new one, regenerate a query
    :param state: graph state for study guide generation
    :return: None
    """
    original_query = state.get("original_query","")
    messages = state.get("messages",[])

    if not original_query:
        # first time: pull the last message from the messages list as the original query
        original_query = messages[-1].content
        prompt = (
            f"""You are an expert at searching over a DOCUMENT. 
            Given an ORIGINAL QUERY from a user your job is to generate a single,
            more specific and detailed search query that will help RETRIEVE CONTEXT
            from the textbook. Focus on clarifying the user's intent and making 
            the NEW QUERY as effective as possible for searching the DOCUMENT.

            The query will be passed to a tool that will retrieve context from the DOCUMENT.
            
            Here is the ORIGINAL QUERY:
            {original_query}
"""
        )

        # this calls the llm with tool capabilities
        new_query = llm_with_tools.invoke(prompt).tool_calls[0]["args"]["subquery"]

        # call tool, with specified new query
        returned_context = query_rag_system.invoke(new_query)

        return {
            "new_query": new_query,
            "retrieved_context": returned_context,
            "original_query": original_query,
            "messages": [new_query],
        }

    else:
        # already generated a query, and need to generate a new one (for after feedback)

        previous_queries = "\n\n".join([f"Attempt {i+1}:\n{q}" for i, q in enumerate(state.get("new_query",[]))])
        previous_retrieved_context = "\n\n".join([f"Attempt {i+1}:\n{ctx}" for i, ctx in enumerate(state.get("retrieved_context",[]))])
        grade = state.get("grade","")

        # get latest feedback
        feedback = state.get("feedback",[-1])

        prompt = f"""
            ORIGINAL QUERY:
            {original_query}

            PREVIOUS QUERIES:
            {previous_queries}

            RETRIEVED CONTEXT:
            {previous_retrieved_context}

            FEEDBACK:
            {feedback}

            GRADE:
            {grade}
            You are an expert educational query generator.
            You will be given an ORIGINAL QUERY which is the question or topic the user wants to study. 
            Please generate a NEW QUERY that is more specific and detailed, based on the ORIGINAL QUERY.
            Based on any feedback or the need for more detail, generate a single new, improved query, to find more or missing information.
"""
        new_query_response = llm_with_tools.invoke(prompt).tool_calls[0]["args"]["query_text"]

        # call tool
        returned_context = query_rag_system.invoke(new_query_response)

        return {
            "new_query": new_query,
            "original_query": original_query,
            "retrieved_context": [returned_context],
            "messages": messages,
        }


def study_guide_generator(state: State, query_rag_system):
    """
    Generates a study guide from the given state of user queries and feedback.
    - if no feedback generates new guide
    - if feedback provided rewrites old guide based on it
    :param state: graph state for study guide generation
    :return: None
    """

    # check if feedback has been given
    study_guide = state.get("study_guide",[])
    feedback = "\n\n".join([f"Attempt {i+1}:\n{f}" for i, f in enumerate(state.get("feedback",[]))])
    grade = state.get("grade","")

    original_query = state.get("original_query","")

    retrieved_context = "\n\n".join([f"Attempt {i+1}:\n{ctx}" for i, ctx in enumerate(state.get("retrieved_context",[]))])

    if grade == None or study_guide == []:
        # study guide needs to be generated
        prompt = f"""
        ORIGINAL QUERY:
        {original_query}
        RETRIEVED CONTEXT:
        {retrieved_context}
        You are an expert educational study guide generator.
        You will be given an ORIGINAL QUERY, which is the question or topic the user wants to study.
        Generate a study guide that is more specific and detailed, based on the ORIGINAL QUERY.
"""
        study_guide = study_guide_gen.invoke(prompt)

        return {"study_guide": study_guide}

    elif state.get("grade") == "Rewrite":
        # rewrite old study guide using feedback
        previous_study_guide = "\n\n".join([f"Attempt {i + 1}:\n{sg}" for i, sg in enumerate(state.get("study_guide", []))])
        retrieved_context = "\n\n".join([f"Attempt {i + 1}:\n{ctx}" for i, ctx in enumerate(state.get("retrieved_context", []))])
        grade = state.get("grade", "")

        # We smash the feedback together separated by attempts
        feedback = "\n\n".join([f"Attempt {i + 1}:\n{f}" for i, f in enumerate(state.get("feedback", []))])

        prompt = f"""
        ORIGINAL QUERY:
        {original_query}
        PREVIOUS STUDY GUIDE:
        {previous_study_guide}
        RETRIEVED CONTEXT:
        {retrieved_context}
        FEEDBACK:
        {feedback}
        current GRADE:
        {grade}

        You are an expert educational study guide generator.
        You've been given an ORIGINAL QUERY, which is the question or topic the user wants to study.
        Please rewrite the PREVIOUS STUDY GUIDE to be more specific and detailed, based on the ORIGINAL QUERY.
"""
        new_study_guide = study_guide_gen.invoke(prompt)
        return {"study_guide": new_study_guide}

    elif state.get("grade") == "More Context":
        # add more context to old study guide
        previous_study_guide = "\n\n".join(
            [f"Attempt {i + 1}:\n{sg}" for i, sg in enumerate(state.get("study_guide", []))])
        retrieved_context = "\n\n".join(
            [f"Attempt {i + 1}:\n{ctx}" for i, ctx in enumerate(state.get("retrieved_context", []))])
        grade = state.get("grade", "")

        # We smash the feedback together separated by attempts
        feedback = "\n\n".join([f"Attempt {i + 1}:\n{f}" for i, f in enumerate(state.get("feedback", []))])

        prompt = f"""
            ORIGINAL QUERY:
            {original_query}
            PREVIOUS STUDY GUIDE:
            {previous_study_guide}
            RETRIEVED CONTEXT:
            {retrieved_context}
            FEEDBACK:
            {feedback}
            current GRADE:
            {grade}

            You are an expert educational study guide generator.
            You've been given an ORIGINAL QUERY, which is the question or topic the user wants to study.
            Please generate a study guide that is more specific,and detailed. 
            Always generate study guides with relevant factual context,and the ORIGINAL QUERY.
    """
        new_study_guide = study_guide_gen.invoke(prompt)

        return {"study_guide": new_study_guide}

    else:
        # the study guide passed
        return {"study_guide": study_guide}



context_grade_description = """
A grade of whether or not the provided study guide is useful for the user's query.

Always use the text within the returned context to grade the study guide. This is direct text from the DOCUMENT.

In order to pass, the study guide MUST meet the following criteria:

1. VARIETY: The guide must use at least 3 different sections of the DOCUMENT, graded by distinct page numbers. If the study guide
is supposed to represent multiple different topics, there must be multiple sections per topic to help student learning. Failures here mean we need more context.

2. RELEVANCE: Each concept represented in the study guide must be pertinent to the user's query. Failures here mean we need more context, or rewriting to better phrase existing concepts.

3. NO HALLUCINATIONS: The study guide must clearly be referencing topics that EXIST in the book. Hallucinated topics are an automatic rewrite.

4. LEARNING PATH:The study guide must provided a clear roadmap to achieving the result of the user's query. If information from the user's query
is not available in the book at all, the guide must instruct the user to find that information elsewhere. Failures here mean we need more context, and possibly a rewrite.

5. CONTEXTUAL KNOWLEDGE ONLY: The study guide MUST only reference information found in the DOCUMENT. The study guide is meant to be a companion for the DOCUMENT, so any 
invention of resources outside of the DOCUMENT is an automatic fail. This usually just means we need a rewrite.

If a guide meets ALL criteria, the grade should be "Pass". Otherwise, the grade will fall into one of the following categories:

"Rewrite": The retrieved context is sufficient for the query, but the study guide needs to be rewritten to meet the criteria.

"More Context": The study guide structure is okay, but the concepts are not relevant to the query. More context needs to be found by searching a new query.

Feedback must include what and why the grade is what it is. Be as strict as necessary. Please choose one of Pass, Rewrite or More Context. 

Iteration are allowed, so prioritize one of the grades if there are multiple possible options.
"""

def feedback_bot(state: State):
    """
    Generates feedback messages for study guide based on criteria,
    depending on feedback routed to study guide gen or query gen.
    :param state: graph state for study guide generation
    :return: None
"""
    # concatenate context seperated by attempts
    original_query = state.get("original_query", "")
    additional_queries = state.get("new_query", [])
    retrieved_context_list = state.get("retrieved_context", [])
    retrieved_context = "\n\n".join([f"Attempt {i + 1}:\n{ctx}" for i, ctx in enumerate(retrieved_context_list)])


    prompt = f'''
    ORIGINAL QUERY:
    {original_query}
    
    STUDY GUIDE:
    {study_guide}

    RETRIEVED CONTEXT:
    {retrieved_context}
    
    ADDITIONAL QUERIES (may be empty):
    {additional_queries}

    CRITERIA:
    {context_grade_description}
    
    You are an expert educational evaluator. You will be provided a STUDY GUIDE,
    which has been constructed from a DOCUMENT. You will also be passed the pages from the DOCUMENT,
    which are being used to justify the readings for the STUDY GUIDE.
    
    You will return Pass, Rewrite, or More Context.

    Rewrites are when no more information is necssary to be found, but the STUDY GUIDE needs more improvement.

    More Context is when the context is not enough to complete the STUDY GUIDE, and the STUDY GUIDE needs more information from our book to be useful.

    Pass is when the STUDY GUIDE meets all CRITERIA.

    Provide your grade, and feedback:
'''
    grade = feedback_llm.invoke(prompt)

    return {"grade": grade.grade, "feedback": [grade.feedback], "num_iterations": state.get("num_iterations", 0) + 1}

def feedback_iteration(state: State):
    '''
    Conditional function to direct state at feedback_bot to end, query, or study guide generator.
    If the feedback bot determines the guide passes, we end.
    If we need a rewrite, we go to the query generator and call.
    If we need more context, we go to the study guide generator.
'''
    grade = state.get("grade", "")

    if grade == "Rewrite":
        return "rewrite"
    elif grade == "More Context":
        return "more_context"
    else:
        return "end"




# st.title("Generate Study Guide :memo:"+":books:")
#     # tool llm
#     llm = ChatNebius(
#         api_key=NEBIUS_API_KEY,
#         model="Qwen/Qwen3-235B-A22B",
#         temperature=0.0,
#     )
#     llm_with_tools = llm.bind_tools([query_rag_system], tool_choice="required")
#
#     if query := st.chat_input("What would you like to generate a study guide for?"):
#         # display user message
#         st.chat_message("user").markdown(query)
#
#         new_query = llm_with_tools.invoke(query).tool_calls[0]["args"]["query_text"]
#
#         # call tool, with specified new query
#         returned_context = query_rag_system.invoke(new_query)
#
#         # display assistant response
#         with st.chat_message("assistant"):
#             st.markdown(returned_context)