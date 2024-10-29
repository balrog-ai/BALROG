# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

APICODEDIR = ../iclbench
APICMD = sphinx-apidoc
APIDOCDIR = api
APIOPTS = ../iclbench/ ../iclbench/agents/ ../iclbench/environments/ ../iclbench/prompt_builder/

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	# rm -rf "$(BUILDDIR)"
	# rm -rf index.md
	# cat ../README.md > index.md
	# cat index_tree.md >> index.md
	@$(APICMD) -e -f -o  "$(APIDOCDIR)" --module-first "$(APICODEDIR)" $(APIOPTS)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
