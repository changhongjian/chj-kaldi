
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

CXXFLAGS += -DKENLM_MAX_ORDER=6 -Ikenlm  \
  -lboost_system-mt -lboost_thread-mt -lz -lbz2   \
   -Wl,-R /usr/lib64 -Wl,-R /usr/local/lib -Wl,-R /lib64 

BINFILES = chj-matrix-max-pdf-to-id chj-beam-search chj-ctc-lstm chj-beam-search-withfst \
             chj-ctc-decode-beam-search-use-int-LM \
			 chj-ctc-lstm-stream chj-decode-ctc-by-pdf chj-decode-ctc-by-shrink-id  

OBJFILES = chj-nnet-loss.o

LIBNAME = chj-static-lib

ADDLIBS = ../nnet/kaldi-nnet.a ../cudamatrix/kaldi-cudamatrix.a ../lat/kaldi-lat.a \
          ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a \
          ../util/kaldi-util.a ../base/kaldi-base.a \
             /data/zyou/wpr/software/kenlm/lib/libkenlm.a \
			  ../_chjcuda/chj-cuda.a

include ../makefiles/default_rules.mk
