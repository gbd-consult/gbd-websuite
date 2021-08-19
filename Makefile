CWD     = $(shell pwd)
ROOT    = $(realpath $(dir $(realpath $(firstword $(MAKEFILE_LIST)))))
DOC     = $(ROOT)/doc
APP     = $(ROOT)/app
INSTALL = $(ROOT)/install

PYTHON = python3

SPHINXOPTS = -v -n -b html -j auto -c $(DOC)/sphinx

.PHONY: help spec client client-dev client-dev-server doc doc-dev-server test image image-debug clean

help:
	@echo ""
	@echo "spec [MANIFEST=<manifest>]               - build the Specs"
	@echo "client [MANIFEST=<manifest>]             - build the Client for production"
	@echo "client-dev [MANIFEST=<manifest>]         - build the Client for development"
	@echo "client-dev-server [MANIFEST=<manifest>]  - start the Client dev server"
	@echo "doc [MANIFEST=<manifest>]                - build the Docs"
	@echo "doc-dev-server [MANIFEST=<manifest>]     - start the Docs dev server"
	@echo "test [MANIFEST=<manifest>]               - run Server tests"
	@echo "image [IMAGE_NAME=<name>]                - build the Docker Image"
	@echo "image-debug [IMAGE_NAME=<name>]          - build the debug Docker Image"
	@echo "clean                                    - remove all build artifacts"
	@echo ""


spec:
	$(PYTHON) $(APP)/gws/spec/generator/run.py build --manifest "$(MANIFEST)"

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

test:
	$(PYTHON) $(APP)/test.py go --manifest "$(MANIFEST)"

image:
	cd $(INSTALL) && $(PYTHON) build.py docker release $(IMAGE_NAME) && cd $(CWD)

image-debug:
	cd $(INSTALL) && $(PYTHON) build.py docker debug $(IMAGE_NAME) && cd $(CWD)

clean:
	find $(ROOT) -name '__build*' -prune -exec rm -rf {} \;
