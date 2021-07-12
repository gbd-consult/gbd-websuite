CWD    = $(shell pwd)
BASE   = $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
DOC    = $(BASE)doc
APP    = $(BASE)app

PYTHON = python3

SPHINXOPTS = -v -n -b html -j auto -c $(DOC)/sphinx

.PHONY: help spec client client-dev client-dev-server doc doc-dev-server image image-debug clean

help:
	@echo ""
	@echo "spec [MANIFEST=<manifest>]               - build the Specs"
	@echo "client [MANIFEST=<manifest>]             - build the Client for production"
	@echo "client-dev [MANIFEST=<manifest>]         - build the Client for development"
	@echo "client-dev-server [MANIFEST=<manifest>]  - start the Client dev server"
	@echo "doc [MANIFEST=<manifest>]                - build the Docs"
	@echo "doc-dev-server [MANIFEST=<manifest>]     - start the Docs dev server"
	@echo "image [IMAGE_NAME=<name>]                - build the Docker Image"
	@echo "image-debug [IMAGE_NAME=<name>]          - build the debug Docker Image"
	@echo ""


spec:
	$(PYTHON) $(APP)/gws/spec/generator/run.py --manifest "$(MANIFEST)"

client-dev: spec
	cd $(APP)/js && npm run dev && cd $(CWD)

client-dev-server: spec
	cd $(APP)/js && npm run dev-server && cd $(CWD)

client: spec
	cd $(APP)/js && npm run production && cd $(CWD)

doc: spec
	$(PYTHON) $(DOC)/sphinx/conf.py pre && \
	sphinx-build -E -a $(SPHINXOPTS) $(DOC)/sphinx $(DOC)/_build && \
	$(PYTHON) $(DOC)/sphinx/conf.py post

doc-dev-server: doc
	sphinx-autobuild -B $(SPHINXOPTS) $(DOC)/sphinx $(DOC)/_build

image:
	$(PYTHON) $(BASE)install/build.py docker release $(IMAGE_NAME) && cd $(CWD)

image-debug:
	$(PYTHON) $(BASE)install/build.py docker debug $(IMAGE_NAME) && cd $(CWD)

clean:
	find $(BASE)app/ -name '__build*'

