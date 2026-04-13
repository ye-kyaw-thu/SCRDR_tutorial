#!/bin/bash

## SCRDR Binary Tree
dot -Tpdf scrdr-binary-tree.dot -o scrdr-binary-tree.pdf
dot -Gdpi=300 -Tpng scrdr-binary-tree.dot -o scrdr-binary-tree.png

## MCRDR N-ary Structure
dot -Tpdf scrdr-binary-tree.dot -o mcrdr-n-ary-structure.pdf
dot -Gdpi=300 -Tpng mcrdr-n-ary-structure.dot -o mcrdr-n-ary-structure.png

## GRDR Chaining Process
dot -Tpdf grdr-chaining-process.dot -o grdr-chaining-process.pdf
dot -Gdpi=300 -Tpng grdr-chaining-process.dot -o grdr-chaining-process.png
