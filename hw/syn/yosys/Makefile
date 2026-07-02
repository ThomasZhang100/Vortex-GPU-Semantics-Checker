ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

SRC_DIR := $(VORTEX_HOME)/hw/syn/yosys

TOP_LEVEL_ENTITY ?= Vortex
PREFIX ?= build

SCRIPT_DIR := $(VORTEX_HOME)/hw/scripts
RTL_DIR := $(VORTEX_HOME)/hw/rtl

CP = cp -rf
RMDIR = rm -rf
ECHO = @echo

BUILD_DIR := $(PREFIX)_$(TOP_LEVEL_ENTITY)
BIN_DIR := $(BUILD_DIR)/bin

# control RTL debug tracing states
DBG_TRACE_FLAGS += -DDBG_TRACE_PIPELINE
DBG_TRACE_FLAGS += -DDBG_TRACE_MEM
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE
DBG_TRACE_FLAGS += -DDBG_TRACE_AFU
DBG_TRACE_FLAGS += -DDBG_TRACE_GBAR
DBG_TRACE_FLAGS += -DDBG_TRACE_TCU

# Control logic analyzer monitors
DBG_SCOPE_FLAGS += -DDBG_SCOPE_AFU
DBG_SCOPE_FLAGS += -DDBG_SCOPE_ISSUE
DBG_SCOPE_FLAGS += -DDBG_SCOPE_FETCH
DBG_SCOPE_FLAGS += -DDBG_SCOPE_LSU

ifdef NUM_CORES
# cluster configuration
CONFIGS_1c  := -DNUM_CLUSTERS=1 -DNUM_CORES=1
CONFIGS_2c  := -DNUM_CLUSTERS=1 -DNUM_CORES=2
CONFIGS_4c  := -DNUM_CLUSTERS=1 -DNUM_CORES=4  -DL2_ENABLE
CONFIGS_8c	:= -DNUM_CLUSTERS=1 -DNUM_CORES=8  -DL2_ENABLE
CONFIGS_16c	:= -DNUM_CLUSTERS=1 -DNUM_CORES=16 -DL2_ENABLE
CONFIGS_32c := -DNUM_CLUSTERS=2 -DNUM_CORES=16 -DL2_ENABLE
CONFIGS_64c := -DNUM_CLUSTERS=4 -DNUM_CORES=16 -DL2_ENABLE
CONFIGS += $(CONFIGS_$(NUM_CORES)c)
endif

# include paths
FPU_INCLUDE = -I$(RTL_DIR)/fpu
ifneq (,$(findstring -DFPU_FPNEW, $(CONFIGS)))
	FPU_INCLUDE += -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src -J$(THIRD_PARTY_DIR)/cvfpu/src/fpu_div_sqrt_mvp/hdl -J$(THIRD_PARTY_DIR)/cvfpu/src
endif
RTL_INCLUDE = -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache
RTL_INCLUDE += $(FPU_INCLUDE)

# Checker (SAURIA systolic array): add sauria dirs so gen_sources copies sa_array,
# sa_processing_element, the FMA wrapper, and the adder/multiplier modules into src/.
# The cvfpu helper files the FMA needs are handled separately (CHK_CVFPU_SRCS below):
# they can't go through gen_sources' whole-dir copy because cvfpu/src/fpnew_fma.sv
# would collide with sauria's own pulp_fma/fpnew_fma.sv.  The -J common_cells/include
# is still required so gen_sources can resolve `include "common_cells/registers.svh"`
# while preprocessing fpnew_fma.sv.
ifneq (,$(findstring -DCHECKER_ENABLE, $(CONFIGS)))
	RTL_INCLUDE += -I$(RTL_DIR)/core/sauria -I$(RTL_DIR)/core/sauria/multiplier -I$(RTL_DIR)/core/sauria/adder -I$(RTL_DIR)/core/sauria/pulp_fma
	RTL_INCLUDE += -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include
	# cvfpu packages/helpers the FMA references — passed to sv2v as explicit sources
	# (none of them have their own `include, so no extra incdirs are needed).
	CHK_CVFPU_SRCS := \
		$(THIRD_PARTY_DIR)/cvfpu/src/fpnew_pkg.sv \
		$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src/cf_math_pkg.sv \
		$(THIRD_PARTY_DIR)/cvfpu/src/fpnew_classifier.sv \
		$(THIRD_PARTY_DIR)/cvfpu/src/fpnew_rounding.sv \
		$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src/lzc.sv
	SV2V_EXTRA := -s "$(CHK_CVFPU_SRCS)"
endif

# Debugging
ifdef DEBUG
	CFLAGS += $(DBG_TRACE_FLAGS)
else
	CFLAGS += -DNDEBUG
endif

# Enable scope analyzer
ifdef SCOPE
	CFLAGS += -DSCOPE $(DBG_SCOPE_FLAGS)
	SCOPE_JSON += $(BUILD_DIR)/scope.json
endif

# Enable perf counters
ifdef PERF
	CFLAGS += -DPERF_ENABLE
endif

CFLAGS += -DSYNTHESIS -DYOSYS
CFLAGS += -DXLEN_$(XLEN)
CFLAGS += $(CONFIGS)
CFLAGS += $(RTL_INCLUDE)

# Build targets
all: clean build

gen-sources: $(BUILD_DIR)/sources.txt
$(BUILD_DIR)/sources.txt:
	mkdir -p $(BUILD_DIR); cd $(BUILD_DIR); $(SCRIPT_DIR)/gen_sources.sh -P $(CFLAGS) -Csrc -Osources.txt

sv2v: $(BUILD_DIR)/project.v
$(BUILD_DIR)/project.v: gen-sources
	cd $(BUILD_DIR); $(SCRIPT_DIR)/sv2v.sh -t$(TOP_LEVEL_ENTITY) -Isrc $(SV2V_EXTRA) -oproject.v

build: $(BUILD_DIR)/project.v
	cd $(BUILD_DIR); $(SRC_DIR)/synth.sh -t$(TOP_LEVEL_ENTITY) -sproject.v

elaborate: $(BUILD_DIR)/project.v
	cd $(BUILD_DIR); $(SRC_DIR)/synth.sh -t$(TOP_LEVEL_ENTITY) -sproject.v -P="elaborate"

# BLACKBOX="mod1 mod2" excludes those modules from synthesis (e.g. all RAM primitives —
#   BLACKBOX="VX_dp_ram VX_sp_ram VX_async_ram_patch" — so SRAM isn't FF-mapped; quote it
#   from a memory-compiler datasheet instead).
# KEEP="mod1 mod2" marks modules so DCE can't strip them (e.g. KEEP=VX_checker to measure a
#   block whose output isn't wired to a top-level pin).
# ALLOW_WARN=1 lets synthesis finish despite Yosys warnings (otherwise treated as errors).
synthesis: $(BUILD_DIR)/project.v
	cd $(BUILD_DIR); $(SRC_DIR)/synth.sh -t$(TOP_LEVEL_ENTITY) -sproject.v -P="synthesis" $(foreach m,$(BLACKBOX),-B$(m)) $(foreach m,$(KEEP),-K$(m)) $(if $(ALLOW_WARN),-W)

clean:
	$(RMDIR) $(BUILD_DIR)
