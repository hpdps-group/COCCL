SHELL := /usr/bin/env bash
MAKEFALGS += -r
.SUFFIXES:
.SECONDARY:

SUBDIRS := $(wildcard */)

all:
	@for dir in $(SUBDIRS); do \
        echo "Creating directory $(SUBOBJDIR)/$$dir"; \
        mkdir -p $(SUBOBJDIR)/$$dir; \
		if [ -f $$dir/Makefile ]; then \
			$(MAKE) -C $$dir SUBOBJDIR=$(SUBOBJDIR); \
		fi; \
	done
