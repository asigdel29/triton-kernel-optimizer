# Local verification and timing commands.
PYTHON ?= python3
.PHONY: check demo bench
check:
	$(PYTHON) -m unittest discover -s tests -v
demo:
	$(PYTHON) -m kernel_optimizer.bench --tokens 17 --repeats 1
bench:
	$(PYTHON) -m kernel_optimizer.bench $(ARGS)

.PHONY: graph
graph:
	bash scripts/graphify.sh
