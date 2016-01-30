SUBDIR	= BandMatrix		\
	  Bezier		\
	  BoxFilter		\
	  BoxFilter2		\
	  DP			\
	  EdgeDetector		\
	  FIRFilter		\
	  GraphCuts		\
	  GuidedFilter		\
	  Heap			\
	  IIRFilter		\
	  List			\
	  NDTree		\
	  PSTree		\
	  Random		\
	  SURF			\
	  Serial		\
	  Sparse		\
	  Spline		\
	  TreeFilter		\
	  Vector		\
	  WeightedMedianFilter	\
	  simd			\
	  stereo1394		\
	  stereoGUI		\
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
