/*
 *  $Id: types.h,v 1.3 2003-02-07 05:14:45 ueshiba Exp $
 */
#ifndef __TUtypes_h
#define __TUtypes_h

#ifdef WIN32
typedef unsigned int	size_t;
typedef unsigned char	u_char;
typedef unsigned short	u_short;
typedef unsigned int	u_int;
typedef unsigned long	u_long;
#else
#  include <sys/types.h>
#endif

typedef signed char		s_char;
typedef long long		int64;
typedef unsigned long long	u_int64;

#ifdef __INTEL_COMPILER
extern "C"
{
    unsigned long long	strtoull(const char*, char**, int);
}
#endif

#endif	/*  !__TUtypes_h	*/
