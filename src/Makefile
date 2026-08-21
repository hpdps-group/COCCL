#
# Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
include ../makefiles/common.mk
include ../makefiles/version.mk
COCCL_MAKEFILE_INCLUDE_ONLY := 1
include coccl-extend/Makefile
COCCL_MAKEFILE_INCLUDE_ONLY :=

##### src files
INCEXPORTS  := nccl.h nccl_net.h
LIBSRCFILES := \
	bootstrap.cc channel.cc collectives.cc debug.cc enqueue.cc group.cc \
	init.cc init_nvtx.cc net.cc proxy.cc transport.cc register.cc\
	$(COCCL_HOST_SRCFILES) \
	$(wildcard graph/*.cc) \
	$(wildcard misc/*.cc) \
	$(wildcard transport/*.cc)

##### lib files
LIBNAME     := libnccl.so
STATICLIBNAME := libnccl_static.a
##### pkgconfig files
PKGCONFIGFILE := nccl.pc
##### dirs
BUILDDIR ?= $(abspath ../build)
INCDIR := $(BUILDDIR)/include
LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj
PKGDIR := $(BUILDDIR)/lib/pkgconfig
##### target files
CUDARTLIB  ?= cudart_static

ifeq ($(CUDARTLIB), cudart_static)
	# Use compatibility shim only with static cudart; see https://github.com/NVIDIA/nccl/issues/658
	LIBSRCFILES += enhcompat.cc
endif

INCTARGETS := $(INCEXPORTS:%=$(INCDIR)/%)
LIBSONAME  := $(LIBNAME:%=%.$(NCCL_MAJOR))
LIBTARGET  := $(LIBNAME:%=%.$(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH))
STATICLIBTARGET := $(STATICLIBNAME)
PKGTARGET  := $(PKGCONFIGFILE)
$(COCCL_CUDA_OBJFILES): $(INCDIR)/nccl.h
LIBOBJ     := $(LIBSRCFILES:%.cc=$(OBJDIR)/%.o) $(COCCL_CUDA_OBJFILES)
DEPFILES   := $(LIBSRCFILES:%.cc=$(OBJDIR)/%.d) $(COCCL_CUDA_DEPFILES)
LDFLAGS    += -L${CUDA_LIB} -l$(CUDARTLIB) -lpthread -lrt -ldl

DEVMANIFEST := $(BUILDDIR)/obj/device/manifest

##### rules
build : $(COCCL_BUILD_TARGET) lib staticlib

lib : $(INCTARGETS) $(LIBDIR)/$(LIBTARGET) $(PKGDIR)/$(PKGTARGET)

staticlib : $(LIBDIR)/$(STATICLIBTARGET)

.PHONY: $(COCCL_BUILD_TARGET)
$(COCCL_BUILD_TARGET): $(INCTARGETS)
	$(MAKE) -f coccl-extend/Makefile $(COCCL_BUILD_TARGET) \
		BUILDDIR=$(BUILDDIR) NCCLDIR=$(abspath ..) COCCL_ROOT=coccl-extend \
		NVCC_GENCODE='$(NVCC_GENCODE)'

$(DEVMANIFEST): ALWAYS_REBUILD $(INCTARGETS)
	$(MAKE) -C ./device

# Empty target to force rebuild
ALWAYS_REBUILD:

-include $(DEPFILES)
$(LIBDIR)/$(LIBTARGET) $(LIBDIR)/$(STATICLIBTARGET) : $(LIBOBJ)

$(INCDIR)/nccl.h : nccl.h.in ../makefiles/version.mk
# NCCL_VERSION(X,Y,Z) ((X) * 10000 + (Y) * 100 + (Z))
	@$(eval NCCL_VERSION := $(shell printf "%d%02d%02d" $(NCCL_MAJOR) $(NCCL_MINOR) $(NCCL_PATCH)))
	mkdir -p $(INCDIR)
	@printf "Generating %-35s > %s\n" $< $@
	sed -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${nccl:Suffix}/$(NCCL_SUFFIX)/g" \
	    -e "s/\$${nccl:Version}/$(NCCL_VERSION)/g" \
	    $< > $@

$(LIBDIR)/$(LIBTARGET): $(LIBOBJ) $(DEVMANIFEST)
	@printf "Linking    %-35s > %s\n" $(LIBTARGET) $@
	mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) -o $@ $(LIBOBJ) $$(cat $(DEVMANIFEST)) $(LDFLAGS)
	ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)

$(LIBDIR)/$(STATICLIBTARGET): $(LIBOBJ) $(DEVMANIFEST)
	@printf "Archiving  %-35s > %s\n" $(STATICLIBTARGET) $@
	mkdir -p $(LIBDIR)
	rm -f $@
	ar cr $@ $(LIBOBJ) $$(cat $(DEVMANIFEST))

$(PKGDIR)/nccl.pc : nccl.pc.in
	mkdir -p $(PKGDIR)
	@printf "Generating %-35s > %s\n" $< $@
	sed -e 's|$${nccl:Prefix}|\$(PREFIX)|g' \
	    -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    $< > $@

$(INCDIR)/%.h : %.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	install -m 644 $< $@

$(INCDIR)/nccl_%.h : include/nccl_%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	install -m 644 $< $@

$(PKGDIR)/%.pc : %.pc
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(PKGDIR)
	install -m 644 $< $@

$(OBJDIR)/%.o : %.cc $(INCTARGETS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CXX) -I. -I$(INCDIR) $(CXXFLAGS) -Iinclude $(COCCL_INCFLAGS) -c $< -o $@
	@$(CXX) -I. -I$(INCDIR) $(CXXFLAGS) -Iinclude $(COCCL_INCFLAGS) -M $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

clean :
	$(MAKE) -C device clean
	rm -rf ${INCDIR} ${LIBDIR} ${PKGDIR} ${OBJDIR}

install : build
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/lib/pkgconfig
	mkdir -p $(PREFIX)/include
	cp -P -v $(BUILDDIR)/lib/lib* $(PREFIX)/lib/
	cp -P -v $(BUILDDIR)/lib/pkgconfig/* $(PREFIX)/lib/pkgconfig/
	cp -v $(BUILDDIR)/include/* $(PREFIX)/include/

FILESTOFORMAT := $(shell find . -name ".\#*" -prune -o \( -name "*.cc" -o -name "*.h" \) -print | grep -v -E 'ibvwrap.h|nvmlwrap.h|gdrwrap.h|nccl.h')
# Note that formatting.mk defines a new target so in order to not overwrite the default target,
# it shouldn't be included at the top. Also, it uses the above definition of FILESTOFORMAT as well
# as the BUILDDIR variable.
include ../makefiles/formatting.mk
