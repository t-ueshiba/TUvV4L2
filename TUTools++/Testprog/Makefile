SUBDIR	= Bezier		\
	  BoxFilter		\
	  EdgeDetector		\
	  GraphCuts		\
	  GuidedFilter		\
	  IIRFilter		\
	  List			\
	  NDTree		\
	  PSTree		\
	  Random		\
	  Serial		\
	  Sparse		\
	  Spline		\
	  Vector

TARGETS	= all clean depend

$(TARGETS):
	@for d in $(SUBDIR); do				\
	  echo "";					\
	  echo "*** Current directory: $$d ***";	\
	  cd $$d;					\
	  $(MAKE) NAME=$$d $@;				\
	  cd ..;					\
	done
