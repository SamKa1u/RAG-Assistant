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
import streamlit as st

class ConceptGuide(BaseModel):
    concept_name: str = Field(description="The name of the concept to be studied")
    concept_description: str = Field(description="A description of the concept to be studied")
    subtopics: list[str] =  Field(description="A detailed list of subtopics relevant to the concept to be studied each entry should have 2 at least sentences")
    cited_pages: list[str] = Field(description="A list of concept relevant pages to be studied")


class StudyGuide(BaseModel):
    original_query: str = Field(description="The original query from the user")
    study_guide: list[ConceptGuide] = Field(
        description="A list of concepts each with several subtopics each having multiple relevant bullet points to be studied by the user all pertaining to the original query")


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
    model="meta-llama/Llama-3.3-70B-Instruct", #Qwen/Qwen3-235B-A22B
    temperature=0.0,
 )
study_guide_gen = study_guide_llm.with_structured_output(StudyGuide)

# feedback llm
feedback_llm_instance = ChatNebius(
    api_key=NEBIUS_API_KEY,
    model="Qwen/Qwen3-4B-fast", #   meta-llama/Meta-Llama-3.1-8B-Instruct
    temperature=0.0,
)
feedback_llm = feedback_llm_instance.with_structured_output(ContextFeedback)

# langchain node definitions
def query_generator_and_call(state: State):
    """
    Handles writing queries and issuing tool all requests.
    if the user has started the chain, use most recent message as the prompt for the query
    if we've already tried to generate a query, and we need a new one, regenerate a query
    :param state: graph state for study guide generation
    :return: None
    """
    st.write("************* query_gen")
    original_query = state.get("original_query","")

    messages = state.get("messages",[])
    if not original_query:
        st.info("*** not original_query")
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
        new_query = llm_with_tools.invoke(prompt).tool_calls[0]["args"]["query_text"]
        st.info("new_query")

        # call tool, with specified new query
        returned_context = query_rag_system.invoke(new_query)
        st.info("___________returned_context")
        return {
            "new_query": [new_query],
            "retrieved_context": [returned_context],
            "original_query": original_query,
            "messages": [new_query],
        }

    else:
        st.info("*** else")
        # already generated a query, and need to generate a new one (for after feedback)
        previous_queries = "\n\n".join([f"Attempt {i+1}:\n{q}" for i, q in enumerate(state.get("new_query",[]))])
        st.info("previous_queries")
        previous_retrieved_context = "\n\n".join([f"Attempt {i+1}:\n{ctx}" for i, ctx in enumerate(state.get("retrieved_context",[]))])
        st.info("previous_retrieved_context")
        grade = state.get("grade","")
        st.info("grade")

        # get latest feedback
        feedback = state.get("feedback",[-1])
        st.info("feedback")

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
        st.info("new_query_response")

        # call tool
        returned_context = query_rag_system.invoke(new_query_response)
        st.info("______________returned_context")
        return {
            "new_query": [new_query_response],
            "original_query": original_query,
            "retrieved_context": [returned_context],
            "messages": messages,
        }


def study_guide_generator(state: State):
    """
    Generates a study guide from the given state of user queries and feedback.
    - if no feedback generates new guide
    - if feedback provided rewrites old guide based on it
    :param state: graph state for study guide generation
    :return: None
    """
    st.info("**************  study_guide_generator")
    # check if feedback has been given
    study_guide = state.get("study_guide",[])
    st.info(study_guide)
    feedback = "\n\n".join([f"Attempt {i+1}:\n{f}" for i, f in enumerate(state.get("feedback",[]))])
    st.info(feedback)
    grade = state.get("grade","")
    st.info(grade)
    original_query = state.get("original_query","")
    st.info("original_query")
    retrieved_context = "\n\n".join([f"Attempt {i+1}:\n{ctx}" for i, ctx in enumerate(state.get("retrieved_context",[]))])
    st.info("retrieved_context")

    if grade == None or study_guide == []:
        st.info("*** grade none")
        # study guide needs to be generated
        prompt = f"""
        ORIGINAL QUERY:
        {original_query}
        RETRIEVED CONTEXT:
        {retrieved_context}
        You are an expert educational study guide generator skilled in concept clarification, question generation, and summary.
        You will be given an ORIGINAL QUERY, which is the question or topic the user wants to study.
        Generate a study guide that is extremely specific and in depth, based on the concepts found in ORIGINAL QUERY. 
        Always populate subtopics with several sentences all relevant and specific to the concept.
        
"""
        # Ask 2 critical thinking questions about the content of each concept.
        study_guide = study_guide_gen.invoke(prompt)
        st.info(study_guide)
        return {"study_guide": study_guide}

    elif state.get("grade") == "Rewrite":

        st.info("*** grade rewrite")
        # rewrite old study guide using feedback
        previous_study_guide = "\n\n".join([f"Attempt {i + 1}:\n{sg}" for i, sg in enumerate(state.get("study_guide", []))])
        st.info("previous_study_guide")
        retrieved_context = "\n\n".join([f"Attempt {i + 1}:\n{ctx}" for i, ctx in enumerate(state.get("retrieved_context", []))])
        st.info("retrieved_context")
        grade = state.get("grade", "")
        st.info(grade)


        # We smash the feedback together separated by attempts
        feedback = "\n\n".join([f"Attempt {i + 1}:\n{f}" for i, f in enumerate(state.get("feedback", []))])
        st.info(feedback)


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

        You are an expert educational study guide generator skilled in concept clarification, question generation, and summary.
        You've been given an ORIGINAL QUERY, which is the question or topic the user wants to study.
        Please rewrite the PREVIOUS STUDY GUIDE to be more specific and detailed, based on the ORIGINAL QUERY. 
        Always populate subtopics with several sentences all relevant and specific to the concept.
"""
        new_study_guide = study_guide_gen.invoke(prompt)
        st.info("________ new_study_guide")

        return {"study_guide": new_study_guide}

    elif state.get("grade") == "More Context":
        st.info("*** grade context")

        # add more context to old study guide
        previous_study_guide = "\n\n".join([f"Attempt {i + 1}:\n{sg}" for i, sg in enumerate(state.get("study_guide", []))])
        st.info("previous_study_guide")

        retrieved_context = "\n\n".join([f"Attempt {i + 1}:\n{ctx}" for i, ctx in enumerate(state.get("retrieved_context", []))])
        st.info("retrieved_context")

        grade = state.get("grade", "")
        st.info(grade)

        # We smash the feedback together separated by attempts
        feedback = "\n\n".join([f"Attempt {i + 1}:\n{f}" for i, f in enumerate(state.get("feedback", []))])
        st.info(feedback)

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

            You are an expert educational study guide generator skilled in concept clarification, question generation, and summary.
            You've been given an ORIGINAL QUERY, which is the question or topic the user wants to study.
            Always increase the level of detail when adding MORE CONTEXT to subtopics. 
            Always generate study guides with relevant factual context,and the ORIGINAL QUERY.
            Always populate subtopics with several sentences all relevant and specific to the concept.
    """
        new_study_guide = study_guide_gen.invoke(prompt)
        st.info("________ new_study_guide")

        return {"study_guide": new_study_guide}

    else:
        st.info("*** study guide pass")

        # the study guide passed
        return {"study_guide": study_guide}



context_grade_description = """
REWRITE: 
The retrieved context is sufficient for the query, but the study guide needs to be rewritten to meet the criteria.

MORE CONTEXT: 
The study guide structure is okay, but the concepts are not relevant to the query. More context needs to be found by searching a new query.

Always use the text within the returned context to grade the study guide. This is direct text from the DOCUMENT.

In order to pass, the study guide MUST meet the following criteria:

1. VARIETY: The study guide must use at least 2 different sections of the DOCUMENT, graded by page numbers. If the study guide
is supposed to represent multiple different topics, there must be multiple subtopics per concept to help student learning. Failures here mean we need MORE CONTEXT.

2. RELEVANCE: Each concept represented in the study guide must be specific to the user's query. Failures here mean we need MORE CONTEXT, or REWRITE to better phrase existing concepts.

3. NO HALLUCINATIONS: The study guide must clearly be referencing topics that EXIST in the book. Hallucinated topics are an automatic REWRITE.

4. LEARNING PATH:The study guide must provide a clear roadmap to achieving the result of the user's query. If information from the user's query
is not available in the DOCUMENT at all, the study guide must instruct the user to find that information elsewhere. Failures here mean we need MORE CONTEXT, and a REWRITE if the learning path is diffiult to follow.

5. CONTEXTUAL KNOWLEDGE ONLY: The study guide MUST only reference information found in the DOCUMENT. The study guide is meant to be a companion for the DOCUMENT, so any 
invention of resources outside of the DOCUMENT is an automatic fail. This usually just means we need a REWRITE.

If a guide meets ALL criteria, the grade should be "Pass". Otherwise, the grade will fall into one of the following categories:

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
    st.info("************** feedback_bot")
    study_guide = state.get("study_guide", "")
    st.info(study_guide)
    # concatenate context seperated by attempts
    original_query = state.get("original_query", "")
    st.info("original_query")
    additional_queries = state.get("new_query", [])
    st.info("additional_queries")
    retrieved_context_list = state.get("retrieved_context", [])
    st.info("retrieved_context_list")
    retrieved_context = "\n\n".join([f"Attempt {i + 1}:\n{ctx}" for i, ctx in enumerate(retrieved_context_list)])
    st.info("retrieved_context")


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

    REWRITES:
    when no more information is necssary to be found, but the STUDY GUIDE needs more improvement.

    More Context:
    when the context is not enough to complete the STUDY GUIDE, and the STUDY GUIDE needs more information from our book to be useful.

    Pass:
    when the STUDY GUIDE meets all CRITERIA.

    Provide your grade, and feedback:
'''
    grade = feedback_llm.invoke(prompt)
    st.info(f"feedback bot:{grade}")
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