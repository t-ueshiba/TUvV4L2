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
#include "TU/Random.h"
#include <cmath>
#ifdef WIN32
#  include "windows/fakeWindows.h"
#else
#  include <sys/time.h>
#endif

namespace TU
{
/************************************************************************
*  class Random								*
************************************************************************/
//! 乱数発生器を生成する．
/*!
   現在時刻をシードとして初期化される．
*/
Random::Random()
    :_mti(N + 1), _hasExtra(false), _extra(0.0)
{
    initialize();
}

//! 現在時刻をシードとして乱数発生器を初期化する．
void
Random::initialize()
{
    timeval tv;
    gettimeofday(&tv, 0);
    initialize(tv.tv_usec);
}
    
//! シードから乱数発生器を初期化する．
/*!
  \param seed	シード
*/
void
Random::initialize(u_long seed)
{
    _mt[0]= seed & 0xffffffffUL;
    for (_mti = 1; _mti < N; _mti++)
    {
        _mt[_mti] = 
	    (1812433253UL * (_mt[_mti-1] ^ (_mt[_mti-1] >> 30)) + _mti); 
      /*
       * See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier.
       * In the previous versions, MSBs of the seed affect
       * only MSBs of the array _mt[].
       * 2002/01/09 modified by Makoto Matsumoto
       */
        _mt[_mti] &= 0xffffffffUL;	// for >32 bit machines
    }
}

//! 複数の初期化キーから乱数発生器を初期化する．
/*!
  \param keys	初期化キーの配列
  \param nkeys	初期化キーの個数
*/
void
Random::initializeByArray(const u_long keys[], int nkeys)
{
    initialize(19650218UL);

    int	i = 1;
    for (int j = 0, k = (N > nkeys ? N : nkeys); k; --k)
    {
        _mt[i] = (_mt[i] ^ ((_mt[i-1] ^ (_mt[i-1] >> 30)) * 1664525UL))
	       + keys[j] + j;		 // non linear
        _mt[i] &= 0xffffffffUL;		 // for WORDSIZE > 32 machines
        ++i;
	++j;
        if (i >= N)
	{
	    _mt[0] = _mt[N-1];
	    i = 1;
	}
        if (j >= nkeys)
	    j = 0;
    }
    for (int k = N - 1; k; k--)
    {
        _mt[i] = (_mt[i] ^ ((_mt[i-1] ^ (_mt[i-1] >> 30)) * 1566083941UL))
	       - i;			// non linear
        _mt[i] &= 0xffffffffUL;		// for WORDSIZE > 32 machines
        ++i;
        if (i >= N)
	{
	    _mt[0] = _mt[N-1];
	    i = 1;
	}
    }

    _mt[0] = 0x80000000UL;	 // MSB is 1; assuring non-zero initial array
}

//! 区間 [0,0xffffffff] 内の32bit長の整数を発生する．
u_long
Random::generateInt32()
{
    const u_long	UPPER_MASK = 0x80000000UL; // most significant w-r bits
    const u_long	LOWER_MASK = 0x7fffffffUL; // least significant r bits
    const u_long	MATRIX_A   = 0x9908b0dfUL; // constant vector a
  // mag01[x] = x * MATRIX_A  for x=0,1
    const u_long	mag01[2]   = {0x0UL, MATRIX_A};

    u_long	y;

    if (_mti >= N)	 // generate N words at one time
    {
        int kk;

        if (_mti == N+1)	// if initialize() has not been called,
            initialize(5489UL);	// a default initial seed is used

        for (kk = 0; kk < N - M; ++kk)
	{
            y = (_mt[kk] & UPPER_MASK) | (_mt[kk+1] & LOWER_MASK);
            _mt[kk] = _mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (; kk < N-1; kk++)
	{
            y = (_mt[kk] & UPPER_MASK) | (_mt[kk+1] & LOWER_MASK);
            _mt[kk] = _mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (_mt[N-1] & UPPER_MASK) | (_mt[0] & LOWER_MASK);
        _mt[N-1] = _mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

        _mti = 0;
    }
  
    y = _mt[_mti++];

  // Tempering
    y ^= (y >> 11);
    y ^= (y <<  7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}

//! 標準ガウス分布に従う実数を発生する．
double
Random::gaussian()
{
    using namespace	std;
    
    if (!_hasExtra)
    {
	double	v0, v1, r;
	do
	{
	    v0 = 2.0 * generateHalfOpenReal() - 1.0;	// -1 <= v0 < 1
	    v1 = 2.0 * generateHalfOpenReal() - 1.0;	// -1 <= v1 < 1
	    r  = v0*v0 + v1*v1;
	} while (r >= 1.0);
	double	fac = sqrt(-2.0 * log(r) / r);
	_extra = v0 * fac;
	_hasExtra = true;
	return v1 * fac;
    }
    else
    {
	_hasExtra = false;
	return _extra;
    }
}
 
}
