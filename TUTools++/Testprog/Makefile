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
	  Vector		\
	  mm			\
	  stereo1394		\
	  stereoGUI		\
	  stereoTime

TARGETS	= all clean depend

$(TARGETS):
	@for d in $(SUBDIR); do				\
	  echo "";					\
	  echo "*** Current directory: $$d ***";	\
	  cd $$d;					\
	  $(MAKE) NAME=$$d $@;				\
	  cd ..;					\
	done
