#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I. -I$(PREFIX)/include

PROGRAM		= matchTest
LIBS		= -lTUTools++ -ltbb
ifneq (,$(findstring darwin,$(OSTYPE)))
  LIBS	       += -framework IOKit -framework CoreFoundation -framework CoreServices
endif

CPPFLAGS	= -DNDEBUG -DUSE_TBB
CFLAGS		= -g
NVCCFLAGS	= -g
ifeq ($(CXX), icpc)
  CFLAGS	= -O3
  NVCCFLAGS	= -O		# -O2以上にするとコンパイルエラーになる．
  CPPFLAGS     += -DSSE3
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CXX)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC .cpp:sC
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/DericheConvolver.h \
		/usr/local/include/TU/Feature.h \
		/usr/local/include/TU/FeatureMatch.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/Heap.h \
		/usr/local/include/TU/ICIA.h \
		/usr/local/include/TU/IIRFilter.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/IntegralImage.h \
		/usr/local/include/TU/Manip.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Random.h \
		/usr/local/include/TU/Ransac.h \
		/usr/local/include/TU/SURFCreator.h \
		/usr/local/include/TU/SeparableFilter2.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/functional.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/mmInstructions.h \
		/usr/local/include/TU/tuple.h \
		/usr/local/include/TU/types.h
HDRS		= MatchImage.h
SRCS		= MatchImage.cc \
		main.cc
OBJS		= MatchImage.o \
		main.o

include $(PROJECT)/lib/p.mk

###
MatchImage.o: MatchImage.h /usr/local/include/TU/Image++.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/FeatureMatch.h \
	/usr/local/include/TU/Random.h /usr/local/include/TU/Ransac.h \
	/usr/local/include/TU/Feature.h /usr/local/include/TU/Manip.h
main.o: /usr/local/include/TU/SURFCreator.h \
	/usr/local/include/TU/Feature.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/Manip.h \
	/usr/local/include/TU/types.h /usr/local/include/TU/IntegralImage.h \
	/usr/local/include/TU/Image++.h /usr/local/include/TU/Heap.h \
	/usr/local/include/TU/FeatureMatch.h /usr/local/include/TU/Random.h \
	/usr/local/include/TU/Ransac.h MatchImage.h \
	/usr/local/include/TU/ICIA.h /usr/local/include/TU/DericheConvolver.h \
	/usr/local/include/TU/IIRFilter.h \
	/usr/local/include/TU/SeparableFilter2.h
