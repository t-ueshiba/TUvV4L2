/*
 *  $Id: functions.h,v 1.2 2006-11-17 01:34:49 ueshiba Exp $
 */
#ifndef __TUfunctions_h
#define __TUfunctions_h

namespace TU
{
template <class T> static inline T	min(T a, T b) {return (a < b ? a : b);}
template <class T> static inline T	max(T a, T b) {return (a > b ? a : b);}
template <class T> static inline T	min(T a, T b, T c)
					{
					    return min(min(a, b), c);
					}
template <class T> static inline T	max(T a, T b, T c)
					{
					    return max(max(a, b), c);
					}
template <class T> static inline T	min(T a, T b, T c, T d)
					{
					    return min(min(a, b, c), d);
					}
template <class T> static inline T	max(T a, T b, T c, T d)
					{
					    return max(max(a, b, c), d);
					}
template <class T> static inline T	diff(T a, T b)
					{
					    return (a > b ? a - b : b - a);
					}
template <class T> static inline T	abs(T a) {return (a > 0 ? a : -a);}
}

#endif	/* __TUfunctions_h */
