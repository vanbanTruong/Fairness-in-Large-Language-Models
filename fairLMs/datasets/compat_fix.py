"""
Compatibility fixes for various libraries used in the bias analysis toolkit.
This module handles compatibility issues between different libraries and versions.
"""

import sys
import importlib.util

def patch_scipy_for_gensim():
    """
    Patch scipy.linalg to add the triu function if it's missing.
    This is needed for gensim 4.3.2 on Python 3.12 where scipy.linalg.triu may not be available.
    """
    import numpy as np
    
    # Check if scipy is installed
    if importlib.util.find_spec("scipy") is None:
        print("scipy is not installed, skipping patch")
        return
    
    # Import scipy and check if linalg.triu exists
    import scipy
    import scipy.linalg
    
    if not hasattr(scipy.linalg, 'triu'):
        def triu_replacement(m, k=0):
            """Replacement for scipy.linalg.triu using numpy"""
            return np.triu(m, k=k)
        
        # Add the replacement function
        scipy.linalg.triu = triu_replacement
        print("Added triu replacement function to scipy.linalg")

def apply_all_fixes():
    """Apply all compatibility fixes"""
    patch_scipy_for_gensim()

# Apply fixes when module is imported
apply_all_fixes() 