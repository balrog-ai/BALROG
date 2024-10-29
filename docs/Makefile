SPHINXBUILD = sphinx-build
SPHINXOPTS = -W
SOURCEDIR = .
BUILDDIR = _build

.PHONY: help clean html

help:
	@echo "Makefile for Sphinx documentation"
	@echo "Usage:"
	@echo "  make html     to build the HTML docs"
	@echo "  make clean    to clean the build directory"

clean:
	rm -rf $(BUILDDIR)/*

html:
	sphinx-apidoc -o ./api ../iclbench
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)
