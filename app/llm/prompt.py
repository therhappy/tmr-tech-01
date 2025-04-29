"""
Module to handle prompt templates and assembly for conversations
"""

import logging
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Based on the RAG template from https://smith.langchain.com/hub/rlm/rag-prompt
template_main_qa_withuncertainty=(
    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, answer with the best guess you can make from the context and add the '_UNSURE:' flag to your answer, \
        followed by why the answer is unsure. Be as concise as possible."
    "Question: {question}"
    "Examples of correct question answering: {qa_context}"
    "Relevant documents: {doc_context}"
    "Relevant tables: {table_context}"
    "Answer:"
) # Version supporting answering with an UNSURE flag, useful for checking the need & use of tools

template_main_qa=(
    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, compute it and answer. If it cannot be computed, answer with the best guess you can make. Be as concise as possible: your answer must be in the form:\
        'NUMBER UNIT (COMPLEMENT)', where NUMBER is the answer, UNIT is the unit of measure and (COMPLENT) is an optional complement to the answer such as the time \
        or the object of the answer if it ambiguous without."
    "Question: {question}"
    "Examples of correct question answering: {qa_context}"
    "Relevant documents: {doc_context}"
    "Relevant tables: {table_context}"
    "Answer:"
) # Target deployment version, no UNSURE flag
prompt_qa = PromptTemplate(input_variables=["question", "qa_context", "doc_context", "table_context"], template=template_main_qa)


def format_1_doc_qacontext(q: str, a: str, gd: str, op: str):
    """"
    Formats a single q-a example to be part of a prompt's context.
    Args:
        q (str): question
        a (str): answer
        gd (str): gold content
        op (str): the operations used
    Returns:
        str: The formatted question context
    """
    contextualized_question = f"""For the similar question: "{q}",
an ideal agent gave the final answer: "{a}"
by reading the best excerpt from documents: "{gd}"
and using the operations: "{op}" ."""
    return contextualized_question


def format_docs_qa_context(docs_qa: list[Document]):
    """
    Formats the documents retrieved for the QA-examples
    Args:
        docs (list[Document]): The documents retrieved from the qa namespace
    Returns:
        str: The formatted context for the prompt
    """
    questions = [d.__dict__["metadata"]["question"] for d in docs_qa]
    answers = [d.__dict__["metadata"]["answer"] for d in docs_qa]
    gold_contents = [d.__dict__["metadata"]["gold_content"] for d in docs_qa]
    operations = [d.__dict__["metadata"]["operations"] for d in docs_qa]
    
    return """\n\n""".join([
        format_1_doc_qacontext(q, a, gd, op)
        for q, a, gd, op in zip(questions, answers, gold_contents, operations)
    ])


def assemble_doc(pre_texts: list[str], post_texts: list[str], table: list[str]):
    """
    Assembles a document with its pre and post text. This is presently a simple concatenation.
    Args:
        pre_texts (list[str]): The pre-texts sentences for the document
        post_texts (list[str]): The post-texts sentences for the document
        table (list[str]): The table content as a list of rows
    Returns:
        str: The assembled document
    """
    pre_block = '\n'.join(pre_texts)
    table = '\n'.join(table)
    post_block = '\n'.join(post_texts)
    return f"{pre_block}\n\n{table}\n\n{post_block}"


def format_docs_context(docs: list[Document]):
    """
    Formats the documents retrieved for the doc context
    Args:
        docs (list[Document]): The documents retrieved from the doc namespace
    Returns:
        str: The formatted context for the prompt
    """
    pre_texts = [d.__dict__["metadata"]["pre_text"] for d in docs]
    post_texts = [d.__dict__["metadata"]["post_text"] for d in docs]
    tables = [d.__dict__["metadata"]["table"] for d in docs]
    ppt = zip(pre_texts, post_texts, tables)
    return """\n\n""".join([assemble_doc(pres, posts, tables) for pres,posts,tables in ppt])


def assemble_table(table_as_rows: list[str]):
    """
    Assembles a table from a list of rows
    Args:
        table (list[list[str]]): The table to assemble
    Returns:
        str: The assembled table
    """
    return "\n".join(table_as_rows)


def format_tables_context(docs_table: list[Document]):
    """
    Formats the documents retrieved for the table documents
    Args:
        docs (list[Document]): The documents retrieved from the table namespace
    Returns:
        str: The formatted context for the prompt
    """
    tables = [d.__dict__["metadata"]["table"] for d in docs_table]
    return """\n\n""".join(assemble_table(table) for table in tables)
