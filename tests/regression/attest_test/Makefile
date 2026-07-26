ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := attest_test

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS    := $(SRC_DIR)/main.cpp
VX_SRCS := $(SRC_DIR)/kernel.cpp

# Default run: valid manifest -> attestation PASS -> GEMM runs.
# Negative cases via -x {sig,magic,kernel,args,sae,thresh,weight}.
OPTS ?= -T8 -H32 -N16 -F8 -t4 -x none

include ../common.mk
