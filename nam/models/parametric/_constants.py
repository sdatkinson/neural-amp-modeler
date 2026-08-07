# Schema version for parametric model files. This is a separate axis from the stock
# MODEL_VERSION in nam.models._constants: parametric files use unique architecture names
# and their own version namespace so the two schemas can iterate independently. Starts at
# major 1 to stay clear of the stock 0.x range and to signal a distinct schema family;
# bump the major on any breaking change to the parametric file schema (param spec layout,
# conditioning layout, etc.).
PARAMETRIC_MODEL_VERSION = "1.0.0"
