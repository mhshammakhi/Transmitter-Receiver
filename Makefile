NVCC      := nvcc
NVCCFLAGS := -std=c++17 -O2 -arch=native
BUILD_DIR   := build

# When called as 'make block DDC', extract 'DDC' as the block name.
BLOCK := $(filter-out block, $(MAKECMDGOALS))

.PHONY: block DDC FilterDownSample BaseBandFilter TimingRecovery clean

# ---- Block dispatcher ----------------------------------------
# Usage: make block <BLOCK_NAME>   e.g.  make block DDC
block:
ifeq ($(BLOCK),)
	@echo "Usage: make block <BLOCK_NAME>"
	@echo "Available blocks: DDC FilterDownSample BaseBandFilter TimingRecovery"
else
	@$(MAKE) --no-print-directory $(BLOCK)
endif

# When 'block' is among the goals, absorb the bare block name so make
# does not try to build it a second time — the block recipe handles it
# via a recursive $(MAKE) call above.
ifeq ($(filter block, $(MAKECMDGOALS)), block)
ifneq ($(BLOCK),)
$(BLOCK):;
endif
else

# ---- DDC block -----------------------------------------------
DDC_SRCS := 01DDC/kernel.cu 01DDC/ddc_main.cu utils/utils.cpp

DDC: | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(DDC_SRCS) -o $(BUILD_DIR)/ddc
	@echo "[DDC] -> $(BUILD_DIR)/ddc"

# ---- FilterDownSample block ----------------------------------
FDS_SRCS := 02FilterDownSample/kernel.cu 02FilterDownSample/fds_main.cu utils/utils.cpp

FilterDownSample: | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -rdc=true $(FDS_SRCS) -lcufft -o $(BUILD_DIR)/filter_downsample
	@echo "[FilterDownSample] -> $(BUILD_DIR)/filter_downsample"

# ---- BaseBandFilter block ------------------------------------
BBF_SRCS := 03BaseBandFilter/kernel.cu 03BaseBandFilter/bbf_main.cu utils/utils.cpp

BaseBandFilter: | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -rdc=true $(BBF_SRCS) -lcufft -o $(BUILD_DIR)/baseband_filter
	@echo "[BaseBandFilter] -> $(BUILD_DIR)/baseband_filter"

# ---- TimingRecovery block ------------------------------------
# Requires demod_parallel.a (Linux rebuild of demod_parallel.cu).
# The repo ships only demod_parallel.lib (Windows COFF) which cannot
# link on Linux.  Build the .a first with:
#   nvcc -std=c++17 -O2 -arch=native -rdc=true -lib \
#        06TimingRecovery/FastGardner/demod_parallel.cu \
#        -o 06TimingRecovery/FastGardner/demod_parallel.a
TR_DIR  := 06TimingRecovery/FastGardner
TR_LIB  := $(TR_DIR)/demod_parallel.a
TR_SRCS := $(TR_DIR)/tr_main.cu utils/utils.cpp

TimingRecovery: | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -rdc=true $(TR_SRCS) $(TR_LIB) -o $(BUILD_DIR)/timing_recovery
	@echo "[TimingRecovery] -> $(BUILD_DIR)/timing_recovery"

endif
# --------------------------------------------------------------

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
