CWD     = $(shell pwd)
BASE    = $(realpath $(dir $(realpath $(firstword $(MAKEFILE_LIST)))))
DOC     = $(BASE)/doc
APP     = $(BASE)/app

SPEC_BUILD = $(APP)/__build

PYTHON = python3

SPHINXOPTS = -v -n -b html -j auto -c $(DOC)/sphinx

.PHONY: help spec client client-help client-dev client-dev-server doc doc-dev-server test image clean

define HELP

GWS Makefile
~~~~~~~~~~~~

	make spec [MANIFEST=<manifest>]
		- build the Specs

	make client [MANIFEST=<manifest>] [ARGS=<builder args>]
		- build the Client for production

	make client-dev [MANIFEST=<manifest>] [ARGS=<builder args>]
		- build the Client for development

	make client-dev-server [MANIFEST=<manifest>] [ARGS=<builder args>]
		- start the Client dev server

	make client-help
		- Client Builder help

	make doc [MANIFEST=<manifest>]
		- build the Docs

	make doc-dev-server [MANIFEST=<manifest>]
		- start the Docs dev server

	make test [MANIFEST=<manifest>]
		- run Server tests

	make image [ARGS=<builder args>]
		- build docker Images

	make image-help
		- image builder help

	make package DIR=<target dir> [MANIFEST=manifest-path]
		- build an Application tarball

	make clean
		- remove all build artifacts

endef

export HELP

help:
	@echo "$$HELP"

mypy:
	cd $(BASE)/app
	mypy .
	cd $(CWD)

spec:
	mkdir -p $(SPEC_BUILD) && $(PYTHON) $(APP)/gws/spec/make.py --out $(SPEC_BUILD) --manifest "$(MANIFEST)" $(ARGS)

client-help:
	cd $(APP)/js && npm run help && cd $(CWD)

client: spec
	cd $(APP)/js && npm run production -- $(ARGS) && cd $(CWD)

client-dev: spec
	cd $(APP)/js && npm run dev && cd $(CWD)

client-dev-server: spec
	cd $(APP)/js && npm run dev-server -- $(ARGS) && cd $(CWD)

doc: spec
	$(PYTHON) $(DOC)/sphinx/conf.py pre && \
	sphinx-build -E -a $(SPHINXOPTS) $(DOC)/sphinx $(DOC)/_build && \
	$(PYTHON) $(DOC)/sphinx/conf.py post

doc-dev-server: doc
	sphinx-autobuild -B $(SPHINXOPTS) $(DOC)/sphinx $(DOC)/_build

test:
	$(PYTHON) $(APP)/test.py go --manifest "$(MANIFEST)"

image:
	$(PYTHON) $(BASE)/install/docker.py $(ARGS)

image-help:
	$(PYTHON) $(BASE)/install/docker.py -h

package:
	$(PYTHON) $(BASE)/install/package.py $(DIR) --manifest $(MANIFEST)

clean:
	find $(BASE) -name '__build*' -prune -exec rm -rfv {} \;
