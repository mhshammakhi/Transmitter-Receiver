NVCC      := nvcc
NVCCFLAGS := -std=c++17 -O2 -arch=native
BUILD_DIR   := build

# When called as 'make block DDC', extract 'DDC' as the block name.
BLOCK := $(filter-out block, $(MAKECMDGOALS))

.PHONY: block DDC clean

# ---- Block dispatcher ----------------------------------------
# Usage: make block <BLOCK_NAME>   e.g.  make block DDC
block:
ifeq ($(BLOCK),)
	@echo "Usage: make block <BLOCK_NAME>"
	@echo "Available blocks: DDC"
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

endif
# --------------------------------------------------------------

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
