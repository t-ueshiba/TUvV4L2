#################################
#  User customizable macros	#
#################################
DEST		= /home/ueshiba/bin
INCDIRS		= -I$(HOME)/include		# header directories

PROGRAM		= listtest
LIBS		= -lTUCollection++ -lTUObject++ -lTUTools++

CFLAGS		= -g
CCFLAGS		= 
LDFLAGS		= -L$(LIBDIR)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= /home/ueshiba/include/TU/Collection++.h \
		/home/ueshiba/include/TU/Object++.h \
		/home/ueshiba/include/TU/types.h
HDRS		= Int.h
SRCS		= Int.cc \
		listtest.cc
OBJS		= Int.o \
		listtest.o

include $(PROJECT)/lib/p.mk
###
Int.o: Int.h /home/ueshiba/include/TU/Object++.h \
	/home/ueshiba/include/TU/types.h
listtest.o: Int.h /home/ueshiba/include/TU/Object++.h \
	/home/ueshiba/include/TU/types.h \
	/home/ueshiba/include/TU/Collection++.h
