import os
import streamlit as st
from config import PINECONE_API_KEY, NEBIUS_API_KEY, SYSTEM_PROMPT, INDEX_NAME
import traceback
import sys
# pdf ingestion and recall
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from chonkie import RecursiveChunker
from pdfminer import high_level

# inference
from openai import OpenAI
from openai.types import embedding_model
from langchain_nebius import ChatNebius


# langchain study guide agent
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import START, END
from graph import *

# initialize pinecone database for string vector embeddings
pc = Pinecone(api_key=PINECONE_API_KEY)
if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec = ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        )
    )
# access specific pinecone index
index = pc.Index(INDEX_NAME)

# initialize pretrained model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# initialize OpenAI-like client for response generation
nebiusClient = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=NEBIUS_API_KEY
)

def get_embedding(text):
    """
    Converts text to high-dimensional vector embeddings
    :param text: text to be converted
    :return: numerical vector representing meaning of input text
    """
    return embedding_model.encode(text)

def process_pdf(file_path):
    """
    Extracts raw text, splits it into meaningful chunks,
    converts to embeddings and then embeds them in Pinecone database
    :param file_path: the location of the pdf file
    """
    text = high_level.extract_text(file_path)

    # divide the text into chunks for better indexing
    chunker = RecursiveChunker()
    chunks = chunker(text)

    # process each chuck converting to embedding
    for i, chunk in enumerate(chunks):
        content = chunk.text
        embedding = get_embedding(content)
        try:
            index.upsert(vectors=[{"id":str(i), "values":1, "metadata":{"text":content}}])
            st.markdown(f":white_check_mark: Chunk {i + 1} successfully upserted")
        except Exception as e:
            st.info(e)
            st.markdown(f"❌ Chunk {i + 1} failed to upsert")




@tool("query_rag_system_tool", parse_docstring=True)
def query_rag_system(query_text):
    """
    Uses a query to find most relevant text chunk from Pinecone database,
    updates openAI-like client with additional context for a more acurate response
    :param query_text:
    :return: AI-generated response
    """
    MSGS = SYSTEM_PROMPT
    query_embedding = get_embedding(query_text)
    q_embedding = query_embedding.tolist()
    # st.markdown(q_embedding)
    # search pincone for best matching chunk
    results = index.query(vector=q_embedding, top_k=1, include_metadata=True)  # - vector: embedding off query
                                                                               # - top_k: number of most similar results
                                                                               # - include_metadata: ensures retrieval of text chunk along with matching embedding
     # sample response: {'matches': [{'id': '5', 'metadata': {'text': 'is a distribution over clean data x given ' 'noisy data zt. The denoiser can then be ' 'used for\n' 'generation e.g., by defining pθ(zt−1|zt) = ' 'P\n' '\n' 'x q(zt−1|zt, x)pθ(x|zt).\n' '\n' 'Architecture Inception Mercury models are ' 'based on a Transformer architecture [40]. ' 'Note\n' 'that this choice of architecture is ' 'orthogonal to the fact that the Mercury ' 'models are diffusion-\n' 'based. Diffusion implies specific training ' 'and generation algorithms, but does not ' 'pose con-\n' 'straints on the architecture of neural ' 'network that is trained. For example, a ' 'dLLM could also\n' 'be based on a recurrent architecture [32, ' '17]. This is analogous to architecture ' 'choices for image\n' 'diffusion models, in which the denoising ' 'network can also be parameterized with a ' 'U-Net [19]\n' 'or a transformer [31]. Relying on a ' 'Transformer architecture has a number of ' 'advantages. It\n' 'allows Mercury models to benefit from ' 'efficient implementations of low-level ' 'primitives, and it\n' 'simplifies hyper-parameter search and ' 'optimization.\n' '\n' 'Fine-tuning and Alignment\n' 'fine-tuning and alignment on downstream ' 'datasets via RLHF [30] or DPO [33] ' 'techniques to\n' 'improve downstream performance. The key ' 'change for all stages is to replace the ' 'autoregressive\n' 'loss with a denoising diffusion loss.\n' '\n' 'Inception Mercury Models can benefit from ' 'further pre-training,\n' '\n' 'Context Length Inception Mercury models ' 'support a context length of up to 32,768 ' 'tokens\n' 'out of the box and up to 128k tokens with ' 'context extension approaches. This ' 'protocol follows\n' 'standard training recipes used for ' 'developing language models [16, 42, 26].\n' '\n' '2.2\n' '\n' 'Inference\n' '\n' 'Prompting In addition to generating full ' 'sequences from scratch, our inference ' 'methods\n' 'support flexible generation conditioned on ' 'a prompt or context. Given that the ' 'Mercury models\n' 'support conditional generation, and given ' 'that they can be trained, fine-tuned, and ' 'aligned\n' 'on datasets that are analogous to those of ' 'traditional language models, the Mercury ' 'models\n' 'also support prompting as in traditional ' 'LLMs. This includes zero-shot prompting, ' 'few-shot\n'}, 'score': 0.446347594, 'values': []}], 'namespace': '', 'usage': {'read_units': 1}}
    # st.markdown(results)

    # extract best matching chunk from pinecone response
    context = results["matches"][0].get("metadata", {}).get("text","")

    # construct prompt with additional context
    prompt = f"DOCUMENT:\n\n {context}\n\nQUESTION:\n\n {query_text}\n\nINSTRUCTIONS:\n\nAnswer the users QUESTION using the DOCOUMENT text above.\nKeep your answer grounded in the the facts of the DOCUMENT.\nIf the DOCUMENT doesn't contain the facts to answer QUESTION then return `NONE`"
    # st.markdown(prompt)

    MSGS.update({"role": "user", "content": prompt})
    # st.markdown(MSGS)

    # get response from model
    try:
        response = nebiusClient.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            messages=[
                MSGS
            ],
            )
    except Exception as e:
        st.info(e)

    # extract and return cleaned response
    return response.choices[0].message.content.strip()


with st.sidebar:
    st.title("RAG Study Assistant")
    st.subheader("", divider=True)
    choice = st.radio("Navigation",["Upsert","Chat","Generate Study Guide"],key="navigation")
    st.markdown("This app allows you to update an LLMs context with your pdf files for better quality responses.")

    if choice == "Generate Study Guide":
        graph_sucess = False
        st.subheader("Study Guide Generator Graph", divider=True)
        # ------ building nodes
        try:
            # query generator and call
            graph_builder.add_edge(START, "query_generator_and_call")
            graph_builder.add_node("query_generator_and_call", query_generator_and_call)
            # study guide generator
            graph_builder.add_node("study_guide_generator", study_guide_generator)
            # feedback bot
            graph_builder.add_node("feedback_bot", feedback_bot)

            # ------ building edges
            # wire query generator to the study guide
            graph_builder.add_edge("query_generator_and_call", "study_guide_generator")

            # wire study guide to feedback bot
            graph_builder.add_edge("study_guide_generator", "feedback_bot")

            # ------ building conditional edges
            # wire feedback bot to study guide generator,and feedback bot to the query call.
            graph_builder.add_conditional_edges(
                "feedback_bot",
                feedback_iteration,
                {
                    "end": END,
                    "rewrite": "study_guide_generator",
                    "more_context": "query_generator_and_call",
                },
            )
        except Exception as e:
            st.info(e)

        try:
            graph = graph_builder.compile()
            st.image(graph.get_graph().draw_mermaid_png(max_retries=5))
            graph_success = True
        except Exception as e:
            st.error(e)

if choice == "Upsert":
    st.title("Upsert to DB :brain:")
    file = st.file_uploader("Upload a PDF to be embedded in the vector database.")
    if file:
        process_pdf(file)

if choice == "Chat":
    st.title("RAG Chatbot:robot:")

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # handle user input
    if prompt := st.chat_input("What would you like to know?"):
        # display user message
        st.chat_message("user").markdown(prompt)
        # add message to history
        st.session_state.messages.append({"role":"user","content":prompt})

        response = query_rag_system(prompt)
        # display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
            # add response to history
            st.session_state.messages.append({"role":"assistant","content":response})

if choice == "Generate Study Guide":
    st.title("Generate Study Guide :memo:"+":books:")
    if graph_success:
        if query := st.chat_input("What would you like to generate a study guide for?"):
            # display user message
            st.chat_message("user").markdown(query)

            msg = HumanMessage(content=query)
            try:
                state = graph.invoke({"messages": [msg]})
            except Exception as e:
                exc_type, exc_value, tb = sys.exc_info()
                st.write("Exception Type:", exc_type.__name__)
                st.write("Exception Value:", exc_value)

                # Print the full traceback
                traceback.print_tb(tb)

            # pretty print the state, get snapshot of output

            # st.write(print("Final output:"))
            #
            # st.write(print("\n=== Original Query ==="))
            # st.write(print(state.get("original_query", "No original query found.")))
            #
            # st.write(print("\n=== Study Guide ==="))
            # resulting_study_guide = state.get("study_guide", "")
            #
            # for sg in resulting_study_guide.study_guide:
            #     st.write(print(f"\n=== Study Guide ==="))
            #     st.write(print(sg))
            #
            # st.write(print("\n=== Retrieved Context ==="))
            # st.write(print(state.get("retrieved_context", "No context retrieved.")))
            #
            # st.write(print("\n=== Grade ==="))
            # st.write(print(state.get("grade", "No grade assigned.")))
            #
            # st.write(print("\n=== Feedback ==="))
            # st.write(print(state.get("feedback", "No feedback provided.")))
            #
            # st.write(print("\n=== Number of Iterations ==="))
            # st.write(print(state.get("num_iterations", "No iterations counted.")))
    else:
        st.error("graph could not be generated")


