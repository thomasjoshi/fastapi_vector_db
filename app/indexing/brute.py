"""
Compatibility module for BruteForceCosine.

This module re-exports LinearSearchCosine as BruteForceCosine for backward compatibility.
"""

from app.indexing.linear_search import LinearSearchCosine

# Re-export LinearSearchCosine as BruteForceCosine for backward compatibility
BruteForceCosine = LinearSearchCosine
