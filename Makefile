BASE = $(shell pwd)

PYTHON = python3

SPHINXOPTS    = -q -n -E -a
SPHINXBASE    = $(BASE)/doc/sphinx
SPHINXBUILD   = $(BASE)/doc/_build

.PHONY: help spec client-dev client doc doc-dev image image-debug clean

help:
	@echo ""
	@echo "commands"
	@echo "--------"
	@echo "spec        - build the Server spec files"
	@echo "client      - build the Client in client/_build"
	@echo "client-dev  - start the Client dev server"
	@echo "doc         - build the Docs in doc/_build"
	@echo "doc-dev     - start the Docs dev server"
	@echo "image       - build the Docker Image (with opt. IMAGE=NAME=...)"
	@echo "image-debug - build the debug Docker Image (with opt. IMAGE=NAME=...)"
	@echo ""

spec:
	$(PYTHON) $(BASE)/specgen/run.py

client-dev: spec
	cd $(BASE)/client && npm run dev-server && cd $(PWD)

client: spec
	cd $(BASE)/client && npm run production && cd $(PWD)

doc: spec
	sphinx-build -b html $(SPHINXOPTS) "$(SPHINXBASE)" $(SPHINXBUILD)

doc-dev: spec
	sphinx-autobuild -B -b html $(SPHINXOPTS) "$(SPHINXBASE)" $(SPHINXBUILD)

image: client
	$(PYTHON) $(BASE)/docker/build.py release $(IMAGE_NAME) && cd $(PWD)

image-debug: client
	$(PYTHON) $(BASE)/docker/build.py debug $(IMAGE_NAME) && cd $(PWD)

clean:
	rm -rf $(BASE)/client/_build
	rm -rf $(BASE)/doc/_build
	rm -rf $(BASE)/docker/_build
