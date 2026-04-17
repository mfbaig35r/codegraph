"""codegraph — Python code relationship graph MCP server."""

import logging

__version__ = "1.1.0"

# Configure library-level logging (NullHandler by default, users can override)
logging.getLogger("codegraph").addHandler(logging.NullHandler())
