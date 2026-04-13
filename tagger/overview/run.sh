#!/bin/bash

# PDF for papers
dot -Tpdf scrdr_tagger_overview.dot -o scrdr_tagger_overview.pdf

# High-resolution PNG for slides
dot -Tpng -Gdpi=300 scrdr_tagger_overview.dot -o scrdr_tagger_overview.png

# SVG (scalable, for LaTeX or web)
dot -Tsvg scrdr_tagger_overview.dot -o scrdr_tagger_overview.svg


