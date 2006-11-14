/*
 *  $Id: functions.h,v 1.1 2006-11-14 06:22:10 ueshiba Exp $
 */
#ifndef __TUfunctions_h
#define __TUfunctions_h

namespace TU
{
template <class T>
static inline T		min(T a, T b)		{return (a < b ? a : b);}
template <class T>
static inline T		max(T a, T b)		{return (a > b ? a : b);}
template <class T>
static inline void	swap(T& a, T& b)	{T tmp = a; a = b; b = tmp;}
}

#endif	/* __TUfunctions_h */
