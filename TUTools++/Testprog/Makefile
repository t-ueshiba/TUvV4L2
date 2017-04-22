SUBDIR	= BandMatrix		\
	  BoxFilter		\
	  BoxFilter2		\
	  DP			\
	  FIRFilter		\
	  GraphCuts		\
	  GuidedFilter		\
	  ICIA			\
	  IIRFilter		\
	  List			\
	  NDTree		\
	  Quantizer		\
	  Random		\
	  Serial		\
	  Sparse		\
	  TreeFilter		\
	  Vector		\
	  WeightedMedianFilter	\
	  array			\
	  tuple			\
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
