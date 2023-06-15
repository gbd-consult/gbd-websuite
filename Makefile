CWD    = $(shell pwd)
BASE   = $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
DOC    = $(BASE)doc

PYTHON = python3

SPHINXOPTS = -v -n -b html -j auto -c $(DOC)/sphinx

.PHONY: help spec client-dev client doc doc-dev image image-debug clean

help:
	echo 'base', $(BASE)
	@echo ""
	@echo "commands"
	@echo "--------"
	@echo "spec             - build the Server spec files"
	@echo "client           - build the Client in client/_build"
	@echo "client-dev       - start the Client dev server"
	@echo "doc              - build the Docs in doc/_build"
	@echo "doc-dev          - start the Docs dev server"
	@echo "image            - build the Docker Image (with optional IMAGE_NAME=...)"
	@echo "image-debug      - build the debug Docker Image (with optional IMAGE_NAME=...)"
	@echo "image-standalone - build the standalone Docker Image (with optional IMAGE_NAME=...)"
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

image: client
	$(PYTHON) $(BASE)install/build.py docker release $(IMAGE_NAME) && cd $(CWD)

image-debug: client
	$(PYTHON) $(BASE)install/build.py docker debug $(IMAGE_NAME) && cd $(CWD)

image-standalone: client
	$(PYTHON) $(BASE)install/build.py docker standalone $(IMAGE_NAME) && cd $(CWD)

clean:
	rm -rf $(BASE)client/_build
	rm -rf $(BASE)doc/_build
	rm -rf $(BASE)install/_build
