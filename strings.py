"""
    List of strings for app
"""

DISCLAIMER = """
It's PoC:
- Do not upload documents with personal or sensitive information
- Data is not stored on the disk and removed after session (close browser or refresh page)
"""

QUERY_EXPLANATION = """
Query is a line in format:

    ColumnName:Query

For example:

    Name:Tell me about document author

- If the list contains more than one request for a specific column name, the application will request it one by one until it receives a response.
- Empty line will be skipped.
- Line started with # will be treated as comment and skipped.
- If ColumnName has prefix "+", it will append response to the previous response for the same column name.
"""