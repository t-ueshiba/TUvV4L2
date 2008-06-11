SUBDIR	= TUTools++		\
	  TUThread++		\
	  TUv++			\
	  TUCalib++		\
	  TUOgl++		\
	  TUSDLOgl++		\
	  TUIeee1394++		\
	  TUObject++		\
	  TUBrep++		\
	  TUCollection++	\
	  TUVision++		\
	  TUXv++		
#	  Kanatani		\
#	  TUXgl++		\
#	  TUXil++		\
#	  TUXilXgl++		\
#	  TUSnapper24++		\
#	  TUS2200++		

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
