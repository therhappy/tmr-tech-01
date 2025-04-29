"""
Tests for the prompt module.
"""

import pytest
from app.llm.prompt import assemble_doc, assemble_table, format_1_doc_qacontext, format_docs_context, format_docs_qa_context, format_tables_context, prompt_qa
from langchain_core.documents import Document

def test_prompt_qa_template():
    """Test that the prompt_qa template is properly defined."""
    # Check that the prompt has the required input variables
    assert "question" in prompt_qa.input_variables
    assert "qa_context" in prompt_qa.input_variables
    assert "doc_context" in prompt_qa.input_variables
    assert "table_context" in prompt_qa.input_variables
    
    # Test the formatting of the prompt
    formatted_prompt = prompt_qa.format(
        question="What is the capital of France?",
        qa_context="France is a country in Europe. Its capital is Paris.",
        doc_context="France is located in Western Europe.",
        table_context="Country | Capital\nFrance | Paris"
    )
    
    assert "What is the capital of France?" in formatted_prompt
    assert "France is a country in Europe. Its capital is Paris." in formatted_prompt
    assert "France is located in Western Europe." in formatted_prompt
    assert "Country | Capital" in formatted_prompt
    assert "France | Paris" in formatted_prompt

def test_format_1_doc_qacontext():
    """Test the formatting of a single QA context."""
    question = "What is machine learning?"
    answer = "Machine learning is a type of artificial intelligence"
    gold_content = "Machine learning is a subset of AI that enables systems to learn and improve from experience"
    operations = "retrieve, analyze, summarize"
    
    formatted_context = format_1_doc_qacontext(question, answer, gold_content, operations)
    
    assert f'For the similar question: "{question}"' in formatted_context
    assert f'an ideal agent gave the final answer: "{answer}"' in formatted_context
    assert f'by reading the best excerpt from documents: "{gold_content}"' in formatted_context
    assert f'and using the operations: "{operations}"' in formatted_context

def test_format_docs_qa_context():
    """Test the formatting of multiple QA contexts from documents."""
    # Create sample documents with metadata
    docs = [
        Document(
            page_content="Sample content 1",
            metadata={
                "question": "What is Python?",
                "answer": "Python is a programming language",
                "gold_content": "Python is a high-level, interpreted programming language",
                "operations": "retrieve, analyze"
            }
        ),
        Document(
            page_content="Sample content 2",
            metadata={
                "question": "What is FastAPI?",
                "answer": "FastAPI is a web framework for Python",
                "gold_content": "FastAPI is a modern web framework for building APIs with Python",
                "operations": "retrieve, summarize"
            }
        )
    ]
    
    # Format the documents
    formatted_context = format_docs_qa_context(docs)
    
    # Check that each document's metadata is included in the formatted context
    assert "For the similar question: \"What is Python?\"" in formatted_context
    assert "an ideal agent gave the final answer: \"Python is a programming language\"" in formatted_context
    assert "by reading the best excerpt from documents: \"Python is a high-level, interpreted programming language\"" in formatted_context
    assert "and using the operations: \"retrieve, analyze\"" in formatted_context
    
    assert "For the similar question: \"What is FastAPI?\"" in formatted_context
    assert "an ideal agent gave the final answer: \"FastAPI is a web framework for Python\"" in formatted_context
    assert "by reading the best excerpt from documents: \"FastAPI is a modern web framework for building APIs with Python\"" in formatted_context
    assert "and using the operations: \"retrieve, summarize\"" in formatted_context
    
    # Check that the formatting produces expected structure
    assert "\n\n" in formatted_context  # Check documents are separated by double newlines

def test_assemble_doc():
    """Test the assembly of a document with pre-text, post-text, and table."""
    pre_texts = ["This is a header.", "This is an introduction."]
    post_texts = ["This is a conclusion.", "This is a footer."]
    table = ["Header1 | Header2", "Value1 | Value2", "Value3 | Value4"]
    
    assembled_doc = assemble_doc(pre_texts, post_texts, table)
    
    # Check that pre-texts are included and properly formatted
    assert "This is a header.\nThis is an introduction." in assembled_doc
    
    # Check that table content is included
    assert "Header1 | Header2\nValue1 | Value2\nValue3 | Value4" in assembled_doc
    
    # Check that post-texts are included and properly formatted
    assert "This is a conclusion.\nThis is a footer." in assembled_doc
    
    # Check overall structure with double new lines between sections
    pre_block = "\n".join(pre_texts)
    table_block = "\n".join(table)
    post_block = "\n".join(post_texts)
    expected = f"{pre_block}\n\n{table_block}\n\n{post_block}"
    assert assembled_doc == expected


def test_format_docs_context():
    """Test the formatting of document contexts."""
    docs = [
        Document(
            page_content="Sample content 1",
            metadata={
                "pre_text": ["Document 1 introduction", "Document 1 context"],
                "post_text": ["Document 1 conclusion", "Document 1 summary"],
                "table": ["Col1 | Col2", "D1V1 | D1V2", "D1V3 | D1V4"]
            }
        ),
        Document(
            page_content="Sample content 2",
            metadata={
                "pre_text": ["Document 2 introduction", "Document 2 context"],
                "post_text": ["Document 2 conclusion", "Document 2 summary"],
                "table": ["Col1 | Col2", "D2V1 | D2V2", "D2V3 | D2V4"]
            }
        )
    ]
    
    formatted_context = format_docs_context(docs)
    
    # Check that both documents are included in the result
    assert "Document 1 introduction\nDocument 1 context" in formatted_context
    assert "Col1 | Col2\nD1V1 | D1V2\nD1V3 | D1V4" in formatted_context
    assert "Document 1 conclusion\nDocument 1 summary" in formatted_context
    
    assert "Document 2 introduction\nDocument 2 context" in formatted_context
    assert "Col1 | Col2\nD2V1 | D2V2\nD2V3 | D2V4" in formatted_context
    assert "Document 2 conclusion\nDocument 2 summary" in formatted_context
    
    # Check that documents are separated by double newlines
    assert "\n\n" in formatted_context


def test_assemble_table():
    """Test the assembly of a table from a list of rows."""
    table_rows = ["Header1 | Header2 | Header3", "Val1 | Val2 | Val3", "Val4 | Val5 | Val6"]
    
    assembled_table = assemble_table(table_rows)
    
    # Check that all rows are included
    assert "Header1 | Header2 | Header3" in assembled_table
    assert "Val1 | Val2 | Val3" in assembled_table
    assert "Val4 | Val5 | Val6" in assembled_table
    
    # Check that rows are properly separated by newlines
    expected = "\n".join(table_rows)
    assert assembled_table == expected


def test_format_tables_context():
    """Test the formatting of table contexts from documents."""
    docs = [
        Document(
            page_content="Sample content 1",
            metadata={
                "table": ["Header1 | Header2", "T1V1 | T1V2", "T1V3 | T1V4"]
            }
        ),
        Document(
            page_content="Sample content 2",
            metadata={
                "table": ["Header1 | Header2", "T2V1 | T2V2", "T2V3 | T2V4"]
            }
        )
    ]
    
    formatted_tables = format_tables_context(docs)
    
    # Check that each table is included in the formatted output
    assert "Header1 | Header2\nT1V1 | T1V2\nT1V3 | T1V4" in formatted_tables
    assert "Header1 | Header2\nT2V1 | T2V2\nT2V3 | T2V4" in formatted_tables
    
    # Check that tables are separated by double newlines
    assert "\n\n" in formatted_tables
