SPHINXOPTS    = -q -n -E -a
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

live:
	@sphinx-autobuild -B -b html $(SPHINXOPTS) "$(SOURCEDIR)" $(BUILDDIR)/html

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
