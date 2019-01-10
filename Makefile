CWD  = $(shell pwd)
BASE = $(dir $(realpath $(firstword $(MAKEFILE_LIST))))



PYTHON = python3

SPHINXOPTS    = -q -n -E -a
SPHINXBASE    = $(BASE)doc/sphinx
SPHINXBUILD   = $(BASE)doc/_build

.PHONY: help spec client-dev client doc doc-dev image image-debug clean

help:
	echo 'base', $(BASE)
	@echo ""
	@echo "commands"
	@echo "--------"
	@echo "spec        - build the Server spec files"
	@echo "client      - build the Client in client/_build"
	@echo "client-dev  - start the Client dev server"
	@echo "doc         - build the Docs in doc/_build"
	@echo "doc-dev     - start the Docs dev server"
	@echo "image       - build the Docker Image (with optional IMAGE_NAME=...)"
	@echo "image-debug - build the debug Docker Image (with optional IMAGE_NAME=...)"
	@echo ""


spec:
	$(PYTHON) $(BASE)specgen/run.py

client-dev: spec
	cd $(BASE)client && npm run dev-server && cd $(CWD)

client: spec
	cd $(BASE)client && \
    npm run production && \
    rm -fr $(BASE)app/web/gws-client && \
    mkdir -p $(BASE)app/web/gws-client && \
    mv $(BASE)client/_build/* $(BASE)app/web/gws-client && \
    cd $(CWD)

doc: spec
	sphinx-build -b html $(SPHINXOPTS) "$(SPHINXBASE)" $(SPHINXBUILD)

doc-dev: spec
	sphinx-autobuild -B -b html $(SPHINXOPTS) "$(SPHINXBASE)" $(SPHINXBUILD)

image: client
	$(PYTHON) $(BASE)docker/build.py release $(IMAGE_NAME) && cd $(CWD)

image-debug: client
	$(PYTHON) $(BASE)docker/build.py debug $(IMAGE_NAME) && cd $(CWD)

clean:
	rm -rf $(BASE)client/_build
	rm -rf $(BASE)doc/_build
	rm -rf $(BASE)docker/_build
