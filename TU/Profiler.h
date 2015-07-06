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
 *  $Id$
 */
/*!
  \file		Profiler.h
  \brief	クラス TU::Profiler の定義と実装
*/
#ifndef __TU_PROFILER_H
#define __TU_PROFILER_H

#include "TU/types.h"
#include "TU/Array++.h"
#include <iostream>
#ifdef WIN32
#  include "windows/fakeWindows.h"
#else
#  include <sys/time.h>
#endif

namespace TU
{
/************************************************************************
*  class Profiler<bool>							*
************************************************************************/
//! プログラムの各ステップ毎に実行時間を測定するためのクラス．
template <bool=true>    class Profiler;
    
template <>
class Profiler<true>
{
  public:
  //! 指定された個数のタイマを持つプロファイラを作成する．
  /*!
    \param ntimers	タイマの個数
  */
    Profiler(size_t ntimers)
	:_active(-1), _accums(ntimers), _t0(), _nframes(0)	{ reset(); }

  //! これまでに処理されたフレーム数を返す．
  /*!
    \return	フレーム数
  */
    size_t		nframes()		const	{ return _nframes; }

    const Profiler&	reset()					const	;
    const Profiler&	start(int n)				const	;

  //! 現在動いているタイマがあれば，それを停止する．
    const Profiler&	stop()			const	{ return start(-1); }

  //! 現在動いているタイマがあればそれを停止し，フレーム番号を一つ進める．
    const Profiler&	nextFrame() const
			{
			    stop();
			    ++_nframes;
			    return *this;
			}

    std::ostream&	print(std::ostream& out)		const	;
    std::ostream&	printTabSeparated(std::ostream& out)	const	;

  private:
    mutable int			_active;
    mutable Array<timeval>	_accums;
    mutable timeval		_t0;
    mutable size_t		_nframes;
};

template <>
struct Profiler<false>
{
    Profiler(size_t)				{}

    void	reset()			const	{}
    void	print(std::ostream&)	const	{}
    void	start(int)		const	{}
    void	nextFrame()		const	{}
};

}
#endif	// !__TU_PROFILER_H
