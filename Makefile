SUBDIR	= TUTools++		\
	  TUv++			\
	  TUOgl++		\
	  TUIIDC++		\
	  TUvIIDC++		\
	  TUXv++		\
	  TUV4L2++		\
	  TUvV4L2++		\
	  TUCuda++		\
	  TUUSB++		\
	  TUHRP2++

TARGETS	= all install clean depend

all:

$(TARGETS):
	@for d in $(SUBDIR); do				\
	  echo "";					\
	  echo "*** Current directory: $$d ***";	\
	  cd $$d;					\
	  $(MAKE) NAME=$$d $@;				\
	  cd ..;					\
	done
