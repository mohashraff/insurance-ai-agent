from smolagents import tool


vectorstore = None


def set_vectorstore(store):
    """
    Makes the current FAISS database available
    to the RAG search tool.
    """

    global vectorstore

    vectorstore = store


@tool
def search_documents(question: str) -> str:
    """
    Searches the uploaded PDF knowledge base for passages
    relevant to a question.

    Use this tool whenever the user asks about information
    contained in the uploaded PDF or asks to compare a claim
    with information in the PDF.

    Args:
        question: The information to search for in the PDF.

    Returns:
        Relevant passages from the uploaded PDF.
    """

    global vectorstore

    if vectorstore is None:
        return "No PDF knowledge base is currently loaded."

    docs = vectorstore.similarity_search(question, k=4)

    if not docs:
        return "The PDF does not provide enough information to answer this."

    passages = []

    for i, doc in enumerate(docs, start=1):

        page = doc.metadata.get("page", "Unknown")

        passages.append(
            f"""
PDF passage {i}
Page: {page}

{doc.page_content}
"""
        )

    return "\n\n".join(passages)