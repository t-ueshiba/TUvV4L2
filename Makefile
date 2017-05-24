SUBDIR	= TUTools++		\
	  TUv++			\
	  TUOgl++		\
	  TUIIDC++		\
	  TUvIIDC++		\
	  TUXv++		\
	  TUV4L2++		\
	  TUvV4L2++		\
	  TUUSB++		\
	  TUHRP2++		\
#	  TUCuda++

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
