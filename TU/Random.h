/*
   A C-program for MT19937, with initialization improved 2002/1/26.
   Coded by Takuji Nishimura and Makoto Matsumoto.

   Before using, initialize the state by using init_genrand(seed)  
   or init_by_array(init_key, key_length).

   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote 
        products derived from this software without specific prior written 
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)

   [Feb.14, 2010] Incorporated into libTUTools++ class library by:
      Toshio UESHIBA
	National Institute of Advanced Industrial Science and Technology (AIST)
   
   $Id$
*/
/*!
  \file		Random.h
  \brief	クラス TU::Random の定義と実装
*/
#ifndef __TURandom_h
#define __TURandom_h

#include "TU/types.h"

namespace TU
{
/************************************************************************
*  class Random								*
************************************************************************/
//! Mersenne Twisterによる乱数発生器
class __PORT Random
{
  public:
    Random()								;

    void	initialize()						;
    void	initialize(u_long seed)					;
    void	initializeByArray(const u_long keys[], int nkeys)	;
    
    u_long	generateInt32()						;
    long	generateInt31()						;
    double	generateClosedReal()					;
    double	generateHalfOpenReal()					;
    double	generateOpenReal()					;
    double	generateRes53()						;
    double	gaussian()						;
    
  private:
    enum	{N = 624, M = 397};

    u_long	_mt[N];
    int		_mti;
    bool	_hasExtra;
    double	_extra;
};
    
//! 区間 [0,0x7fffffff] 内の31bit長の整数を発生する．
inline long
Random::generateInt31()
{
    return (long)(generateInt32() >> 1);
}

//! 区間 [0,1] 内の実数を発生する．
inline double
Random::generateClosedReal()
{
    return generateInt32()*(1.0/4294967295.0);     // divided by 2^32-1
}

//! 区間 [0,1) 内の実数を発生する．
inline double
Random::generateHalfOpenReal()
{
    return generateInt32()*(1.0/4294967296.0);     // divided by 2^32
}

//! 区間 (0,1) 内の実数を発生する．
inline double
Random::generateOpenReal()
{
  // divided by 2^32
    return (((double)generateInt32()) + 0.5)*(1.0/4294967296.0);
}

//! 区間 [0,1) 内の53bit長の実数を発生する．
inline double
Random::generateRes53() 
{ 
    u_long	a = generateInt32() >> 5,
		b = generateInt32() >> 6; 
    return(a*67108864.0 + b)*(1.0/9007199254740992.0); 
} 

}

#endif	// !__TURandom_h
