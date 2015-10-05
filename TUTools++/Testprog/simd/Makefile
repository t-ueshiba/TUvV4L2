SUBDIR	= cvt_iterator_test		\
	  cvt_mask_iterator_test	\
	  cvt_mask_test			\
	  shuffle_test

TARGETS	= all clean depend

$(TARGETS):
	@for d in $(SUBDIR); do				\
	  echo "";					\
	  echo "*** Current directory: $$d ***";	\
	  cd $$d;					\
	  $(MAKE) NAME=$$d $@;				\
	  cd ..;					\
	done
