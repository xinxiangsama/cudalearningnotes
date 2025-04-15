# Compiler and flags
NVCC = nvcc
CXXFLAGS = -std=c++11 -O2 -Wno-deprecated-gpu-targets -I/usr/local/cuda-12.8/include
# LDFLAGS
# Target executable
TARGET = matrixmul

# Source files
SRCS = $(wildcard *.cu)
OBJS = $(SRCS:.cu=.o)

# Default rule
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJS)
	$(NVCC) $(CXXFLAGS) -o $@ $^

# Compile .cu files to .o files
%.o: %.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean