import logging
import warnings
# from PyPDF2.errors import PdfReadWarning # Keep if you want to add PyPDF2 specific warning filters here later

def apply_global_test_configurations():
    """Applies global configurations for tests."""
    print("Applying global test configurations from global_test_config.py...")

    # Suppress pdfminer.pdfpage CropBox log messages (which are warnings)
    logging.getLogger('pdfminer.pdfpage').setLevel(logging.ERROR)
    logging.getLogger('pdfminer').setLevel(logging.ERROR) # Also set for the parent pdfminer logger

    # Example for future: Suppress a specific PyPDF2 warning if needed
    # warnings.filterwarnings("ignore", category=PdfReadWarning, message="Specific PyPDF2 warning to ignore")

    # Example for future: Suppress other warnings by category or message
    # warnings.filterwarnings("ignore", category=UserWarning, module="some_library_module")

    print("Global test configurations applied.")

# If you want this to apply automatically when this module is imported by a test script,
# you could call it here. However, explicit calls from test scripts or conftest.py are often clearer.
# apply_global_test_configurations() 