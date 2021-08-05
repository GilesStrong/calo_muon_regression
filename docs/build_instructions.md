# Build instructions

1. Run: sphinx-apidoc -l -e -M -o source/ ../muon_regression ../muon_regression/basics.py -f
1. Change toctree max depth to 1 in source files
1. Ensure TOC in source/index.rst is up to date
1. Run: make html
1. Open build/html/index.html
