ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := checker_test

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS    := $(SRC_DIR)/main.cpp
VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?= -T8 -H64 -N16 -F32 -t4

include ../common.mk
