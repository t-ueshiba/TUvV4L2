/*
 *  $Id: types.h,v 1.1.1.1 2002-07-25 02:14:16 ueshiba Exp $
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

#endif	/*  !__TUtypes_h	*/
