SUBDIR	= cvt_iterator			\
	  cvt_iterator_test		\
	  cvt_mask_iterator		\
	  cvt_mask_iterator_test	\
	  cvt_pack			\
	  dup				\
	  lookup			\
	  transform

TARGETS	= all clean depend

$(TARGETS):
	@for d in $(SUBDIR); do				\
	  echo "";					\
	  echo "*** Current directory: $$d ***";	\
	  cd $$d;					\
	  $(MAKE) NAME=$$d $@;				\
	  cd ..;					\
	done
