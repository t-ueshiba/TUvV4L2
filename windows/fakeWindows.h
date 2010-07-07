/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *  
 *  $Id: fakeWindows.h,v 1.4 2010-07-07 06:23:21 ueshiba Exp $
 */
#ifndef __FAKEWINDOWS_H
#define __FAKEWINDOWS_H

#ifdef WIN32
#include "TU/types.h"
#include <winsock2.h>

#undef RGB
#undef min
#undef max

#ifdef  __cplusplus
extern "C" {
#endif

/************************************************************************
*  getopt()								*
************************************************************************/
__PORT extern char*	optarg;
__PORT extern int	optind;
__PORT extern int	opterr;
__PORT extern int	optopt;

__PORT int  getopt(int argc, char* const* argv, const char* optstring)	;

/* Describe the long-named options requested by the application.
   The LONG_OPTIONS argument to getopt_long or getopt_long_only is a vector
   of `struct option' terminated by an element containing a name which is
   zero.

   The field `has_arg' is:
   no_argument          (or 0) if the option does not take an argument,
   required_argument    (or 1) if the option requires an argument,
   optional_argument    (or 2) if the option takes an optional argument.

   If the field `flag' is not NULL, it points to a variable that is set
   to the value given in the field `val' when the option is found, but
   left unchanged if the option is not found.

   To have a long-named option do something other than set an `int' to
   a compiled-in constant, such as set a value from `optarg', set the
   option's `flag' field to zero and its `val' field to a nonzero
   value (the equivalent single-letter option character, if there is
   one).  For long options that have a zero `flag' field, `getopt'
   returns the contents of the `val' field.  */

struct option
{
# if (defined __STDC__ && __STDC__) || defined __cplusplus
  const char*	name;
# else
  char*		name;
# endif
  /* has_arg can't be an enum because some compilers complain about
     type mismatches in all the code that assumes it is an int.  */
  int		has_arg;
  int*		flag;
  int		val;
};

/* Names for the values of the `has_arg' field of `struct option'.  */
# define no_argument            0
# define required_argument      1
# define optional_argument      2

__PORT int  getopt_long(int ___argc, char* const* ___argv,
			const char* __shortopts,
			const struct option* __longopts, int* __longind);
__PORT int  getopt_long_only(int ___argc, char* const* ___argv,
			     const char* __shortopts,
			     const struct option* __longopts,
			     int* __longind)				;

/************************************************************************
*  gettimeofday()							*
************************************************************************/
typedef struct timeZone
{
    int		tz_minuteswest;
    int		tz_dsttime;
} timeZone;

__PORT int	gettimeofday(struct timeval* tv, struct timeZone* tz)	;

/************************************************************************
*  srand48(), lrand48(), erand48(), drand48()				*
************************************************************************/
__PORT void	srand48(long seed)					;
__PORT long	lrand48()						;
__PORT double	erand48(unsigned short xseed[3])			;
__PORT double	drand48()						;

/************************************************************************
*  usleep()								*
************************************************************************/
__PORT int	usleep(unsigned int)					;

#ifdef  __cplusplus
}
#endif

#endif	/* WIN32		*/
#endif	/* !__FAKEWINDOWS_H	*/
