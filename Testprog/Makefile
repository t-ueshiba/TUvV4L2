SUBDIR	= BandMatrix		\
	  Bezier		\
	  BoxFilter		\
	  BoxFilter2		\
	  DP			\
	  EdgeDetector		\
	  FIRFilter		\
	  GraphCuts		\
	  GuidedFilter		\
	  ICIA			\
	  IIRFilter		\
	  List			\
	  Mesh			\
	  NDTree		\
	  Quantizer		\
	  Serial		\
	  Sparse		\
	  Spline		\
	  SURF			\
	  TreeFilter		\
	  Vector		\
	  WeightedMedianFilter	\
	  array			\
	  filterStereo		\
	  pair			\
	  tuple			\
	  simd			\
	  stereoGUI		\
	  stereoIIDC		\
	  stereoTest		\
	  stereoTime		\
	  stereoV4L2

TARGETS	= all clean depend

$(TARGETS):
	@for d in $(SUBDIR); do				\
	  echo "";					\
	  echo "*** Current directory: $$d ***";	\
	  cd $$d;					\
	  $(MAKE) NAME=$$d $@;				\
	  cd ..;					\
	done
