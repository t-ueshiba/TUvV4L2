SUBDIR	= TUTools++		\
	  TUv++			\
	  TUOgl++		\
	  TUIeee1394++		\
	  TUObject++		\
	  TUBrep++		\
	  TUCollection++	\
	  TUVision++		\
	  TUXv++		\
	  TUCuda++		\
	  TUV4L2++		\
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
