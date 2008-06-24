/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，第三者へ開示する等の著
 *  作権を侵害する行為を禁止します．
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
 *  without permission by the creator are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holders or the creator are not responsible for any
 *  damages in the use of this program.
 *  
 *  $Id: Profiler.h,v 1.1 2008-06-24 00:40:50 ueshiba Exp $
 */
#include <iostream>
#include <sys/time.h>
#include "TU/Array++.h"

namespace TU
{
/************************************************************************
*  clsss Profiler							*
************************************************************************/
//! プログラムの各ステップ毎に実行時間を測定するためのクラス．
class Profiler
{
  private:
    class Timer
    {
      public:
	Timer()							{reset();}

	Timer&	reset()						;
	Timer&	start()						;
	Timer&	stop()						;
	timeval	print(std::ostream& out, u_int nframes)	const	;
	
      private:
	timeval	_accum;
	timeval	_t0;
    };

  public:
  //! 指定された個数のタイマを持つプロファイラを作成する．
  /*!
    \param ntimers	タイマの個数
   */
    Profiler(u_int ntimers)
	:_active(0), _timers(ntimers), _nframes(0)		{}

    u_int		nframes()			const	;
    const Profiler&	reset()				const	;
    const Profiler&	start(int n)			const	;
    const Profiler&	stop()				const	;
    const Profiler&	nextFrame()			const	;
    std::ostream&	print(std::ostream& out)	const	;

  private:
    mutable int			_active;
    mutable Array<Timer>	_timers;
    mutable u_int		_nframes;
};
    
//! これまでに処理されたフレーム数を返す．
/*!
  \return	フレーム数
 */
inline u_int
Profiler::nframes() const
{
    return _nframes;
}

//! 指定されたタイマを起動する．
/*!
  \param n	タイマの番号
 */
inline const Profiler&
Profiler::start(int n) const
{
    _active = n;
    _timers[_active].start();
    return *this;
}

//! 現在起動中のタイマを停止する．
inline const Profiler&
Profiler::stop() const
{
    _timers[_active].stop();
    return *this;
}

//! フレーム番号を一つ進める．
inline const Profiler&
Profiler::nextFrame() const
{
    ++_nframes;
    return *this;
}

inline Profiler::Timer&
Profiler::Timer::reset()
{
    _accum.tv_sec = _accum.tv_usec = 0;
    _t0.tv_sec = _t0.tv_usec = 0;
    return *this;
}

inline Profiler::Timer&
Profiler::Timer::start()
{
    gettimeofday(&_t0, NULL);
    return *this;
}
    
inline Profiler::Timer&
Profiler::Timer::stop()
{
    timeval	t1;
    gettimeofday(&t1, NULL);
    _accum.tv_sec  += (t1.tv_sec  - _t0.tv_sec);
    _accum.tv_usec += (t1.tv_usec - _t0.tv_usec);
    return *this;
}

}
