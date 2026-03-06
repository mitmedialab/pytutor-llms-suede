def sse(data: str, event: str = "message"):
    """Format a string as an SSE (Server Side Event) payload"""
    data_formatted = data.replace("\r", "").replace("\n", "\ndata: ")
    return f"event: {event}\ndata: {data_formatted}\n\n"
