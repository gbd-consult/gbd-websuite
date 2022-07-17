CWD    = $(shell pwd)
BASE   = $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
DOC    = $(BASE)doc

PYTHON = python3

SPHINXOPTS = -v -n -b html -j auto -c $(DOC)/sphinx

.PHONY: help spec client-dev client doc doc-dev clean

help:
	@echo ""
	@echo "spec        - build the Server spec files"
	@echo "client      - build the Client"
	@echo "client-dev  - start the Client dev server"
	@echo "doc         - build the Docs"
	@echo "doc-dev     - start the Docs dev server"
	@echo "package DIR=<dir> [MANIFEST=manifest-path] - build the Application package"
	@echo ""


spec:
	$(PYTHON) $(BASE)specgen/run.py

client-dev: spec
	cd $(BASE)client && npm run dev-server && cd $(CWD)

client: spec
	cd $(BASE)client && \
	npm run production && \
	rm -fr $(BASE)app/www/gws-client && \
	mkdir -p $(BASE)app/www/gws-client && \
	mv $(BASE)client/_build/* $(BASE)app/www/gws-client && \
	cd $(CWD)

doc: spec
	$(PYTHON) $(DOC)/sphinx/conf.py pre && \
	sphinx-build -E -a $(SPHINXOPTS) $(DOC)/sphinx $(DOC)/_build && \
	$(PYTHON) $(DOC)/sphinx/conf.py post

doc-dev: doc
	sphinx-autobuild --open-browser $(SPHINXOPTS) $(DOC)/sphinx $(DOC)/_build

package:
	$(PYTHON) $(BASE)install/package.py $(DIR) --manifest $(MANIFEST)

clean:
	rm -rf $(BASE)client/_build
	rm -rf $(BASE)doc/_build
	rm -rf $(BASE)install/___build
