SUBDIR	= BandMatrix		\
	  BoxFilter		\
	  BoxFilter2		\
	  DP			\
	  FIRFilter		\
	  GraphCuts		\
	  IIRFilter		\
	  List			\
	  NDTree		\
	  Quantizer		\
	  Random		\
	  Serial		\
	  Sparse		\
	  TreeFilter		\
	  Vector		\
	  simd

TARGETS	= all clean depend

$(TARGETS):
	@for d in $(SUBDIR); do				\
	  echo "";					\
	  echo "*** Current directory: $$d ***";	\
	  cd $$d;					\
	  $(MAKE) NAME=$$d $@;				\
	  cd ..;					\
	done
