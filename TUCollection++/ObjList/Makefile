#################################
#  User customizable macros	#
#################################
DEST		= /home/ueshiba/bin
INCDIRS		= -I../../TUObject++ -I$(HOME)/include

PROGRAM		= listtest
LIBS		= -lTUCollection++ -lTUObject++

CPPFLAGS	= -DTUObjectPP_DEBUG
CFLAGS		= -g
CCFLAGS		= -g
LDFLAGS		= -L$(LIBDIR)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC
EXTHDRS		= ../../TUObject++/Object++_.h \
		../../TUObject++/TU/Object++.h \
		/Users/ueshiba/include/TU/Collection++.cc \
		/Users/ueshiba/include/TU/Collection++.h \
		/Users/ueshiba/include/TU/types.h
HDRS		= Int.h
SRCS		= Desc.cc \
		Int.cc \
		listtest.cc
OBJS		= Desc.o \
		Int.o \
		listtest.o

include $(PROJECT)/lib/p.mk
###
Desc.o: ../../TUObject++/Object++_.h ../../TUObject++/TU/Object++.h \
	/Users/ueshiba/include/TU/types.h
Int.o: Int.h ../../TUObject++/TU/Object++.h \
	/Users/ueshiba/include/TU/types.h
listtest.o: Int.h ../../TUObject++/TU/Object++.h \
	/Users/ueshiba/include/TU/types.h \
	/Users/ueshiba/include/TU/Collection++.h \
	/Users/ueshiba/include/TU/Collection++.cc
