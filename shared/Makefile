
CXX = g++
LD = g++


# enable providers as needed
CUFFT_FLAG = 0
FFTW_FLAG  = 1
MKL_FLAG   = 0



############################################################################

EXE 		:= fftime
BINDIR		:= bin
ROOTOBJDIR	:= obj
SRCDIR		:= src

verbose := 0

debug   := 0
release := 1
profile := 0


ifeq ($(debug), 1)
	BUILD_TYPE 	:= debug
else 
ifeq ($(profile), 1)
	BUILD_TYPE 	:= profile
else 
ifeq ($(release), 1)
	BUILD_TYPE 	:= release
endif
endif
endif

ifeq ($(verbose), 1)
	VERBOSE :=
else
	VERBOSE := @
endif

ifeq ($(BUILD_TYPE), debug)
	CXXFLAGS += -g -DDEBUG  
       	SUFFIX	:= _dbg 	
endif

ifeq ($(BUILD_TYPE),release)
	CXXFLAGS += -O2   
  SUFFIX	:=  	
	CXXFLAGS +=  
endif

ifeq ($(BUILD_TYPE),profile)
	CXXFLAGS += -g -pg
	LFLAGS += -pg
  SUFFIX	:= _prof 	
endif

CXXFLAGS += -Wall -DPRECISIONCHECK 

TARGET	:= $(BINDIR)/$(EXE)$(SUFFIX)
OBJDIR	:= $(ROOTOBJDIR)/$(BUILD_TYPE)
SRCS	:= $(notdir $(wildcard $(SRCDIR)/*.cpp))

CXXFLAGS += -I $(SRCDIR)
OMP_ENABLE = 0

# FFTW related stuff
ifeq ($(FFTW_FLAG), 1)
ifdef FFTW_INCLUDE
CXXFLAGS += -I$(FFTW_INCLUDE)  
endif
ifdef FFTW_LIB
LFLAGS += -L$(FFTW_LIB) 
endif
LFLAGS   += -lfftw3_threads -lfftw3f_threads -lfftw3 -lfftw3f -lpthread -lm -fopenmp
SRCS	 += $(patsubst $(SRCDIR)%,%,$(wildcard $(SRCDIR)/FFTW/*.cpp))
CXXFLAGS += -I $(SRCDIR)/FFTW -fopenmp -DENABLEFFTW
OMP_ENABLE = 1
endif

# CUFFT related stuff
ifeq ($(CUFFT_FLAG), 1)
ifdef CUDA_INCLUDE
CXXFLAGS += -I$(CUDA_INCLUDE)  
endif
ifdef CUDA_LIB
LFLAGS += -L$(CUDA_LIB) 
endif
LFLAGS   += -lcufft -lcuda -lcudart 
SRCS	 += $(patsubst $(SRCDIR)%,%,$(wildcard $(SRCDIR)/CUFFT/*.cpp))
CXXFLAGS += -I $(SRCDIR)/CUFFT -DENABLECUFFT
endif

# MKL related stuff
ifeq ($(MKL_FLAG), 1)
CXXFLAGS += -I /opt/intel/composer_xe_2011_sp1.6.233/mkl/include 
LFLAGS   += -L /opt/intel/composer_xe_2011_sp1.6.233/mkl/lib/intel64/
#LFLAGS   += -Wl,--start-group -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -Wl,--end-group 
LFLAGS   += -Wl,--start-group -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -Wl,--end-group \
						-liomp5
SRCS	   += $(patsubst $(SRCDIR)%,%,$(wildcard $(SRCDIR)/MKL/*.cpp))
CXXFLAGS += -I $(SRCDIR)/MKL -openmp -DENABLEMKL
OMP_ENABLE = 1
endif

ifeq ($(OMP_ENABLE), 1)
LFLAGS += -lpthread -lm -fopenmp
endif

HEADERS	:= $(wildcard $(SRCDIR)/*.h)
OBJS	:= $(patsubst %.cpp,$(OBJDIR)/%.o,$(SRCS))


.PHONY: clean backup

all: makedirs $(TARGET)

# C++ source
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@echo CPP compiling \($(BUILD_TYPE)\) $@
	$(VERBOSE)$(CXX) -c $(CXXFLAGS) -o $@ $<

# linking
$(TARGET): $(OBJS)
	@echo linking \($(BUILD_TYPE)\) $@
	$(VERBOSE)$(LD) -o $@ $^ $(LFLAGS)

makedirs:
	$(VERBOSE)mkdir -p $(OBJDIR)
ifeq ($(CUFFT_FLAG), 1)
	$(VERBOSE)mkdir -p $(OBJDIR)/CUFFT
endif
ifeq ($(FFTW_FLAG), 1)
	$(VERBOSE)mkdir -p $(OBJDIR)/FFTW
endif
ifeq ($(MKL_FLAG), 1)
	$(VERBOSE)mkdir -p $(OBJDIR)/MKL
endif
	$(VERBOSE)mkdir -p $(BINDIR)

clean:
	@echo cleaning project...
	$(VERBOSE)rm -rf $(OBJDIR)/* $(TARGET)

cleanall:
	@echo deep cleaning
	$(VERBOSE)rm -rf $(ROOTOBJDIR)/* 
	$(VERBOSE)rm -rf $(BINDIR)/* 
	$(VERBOSE)rmdir $(BINDIR) 
	$(VERBOSE)rmdir $(ROOTOBJDIR)

doc:
	@echo creating html documentation...
	$(VERBOSE)doxygen fftbench.doxy > log.doxygen 2>&1

# DEPENDENCIES

$(OBJDIR)/main.o: $(HEADERS)

