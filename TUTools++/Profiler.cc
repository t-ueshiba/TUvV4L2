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
#define PROFILE

#include "TU/Profiler.h"
#include <iomanip>

namespace TU
{
/************************************************************************
*  clsss Profiler							*
************************************************************************/
//! 全てのタイマをリセットする（蓄積時間を空にし，フレーム番号を0に戻す）．
const Profiler<true>&
Profiler<true>::reset() const
{
    _active = -1;
    for (size_t n = 0; n < _accums.size(); ++n)
	_accums[n].tv_sec = _accums[n].tv_usec = 0;
    _t0.tv_sec = _t0.tv_usec = 0;
    _nframes = 0;
    return *this;
}

//! 現在動いているタイマがあればそれを停止し，指定されたタイマを起動する．
/*!
  \param n	タイマの番号
 */
const Profiler<true>&
Profiler<true>::start(int n) const
{
    if (n != _active)
    {
	if (_active >= 0)	// 稼働中のタイマがあれば...
	{			// これまでの稼働時間を加算して停止する．
	  // 起動時刻から現在までの時間を稼働中のタイマに加算
	    timeval	t1;
	    gettimeofday(&t1, NULL);
	    _accums[_active].tv_sec  += (t1.tv_sec  - _t0.tv_sec);
	    _accums[_active].tv_usec += (t1.tv_usec - _t0.tv_usec);
	    _active = -1;		// タイマを停止
	}
	if (0 <= n && n < _accums.size())
	{
	    gettimeofday(&_t0, NULL);	// 起動時刻を記録
	    _active = n;		// タイマを起動
	}
    }

    return *this;
}

//! 1フレームあたりの実行時間と1秒あたりの処理フレーム数を表示する．
/*!
  処理速度は，各タイマ毎の蓄積時間から計算されたものと，全タイマの蓄積時間の
  総計から計算されたものの両方が表示される．
  \param out	出力ストリーム
 */
std::ostream&
Profiler<true>::print(std::ostream& out) const
{
    using namespace	std;
    
    timeval		total;
    total.tv_sec = total.tv_usec = 0;

    for (size_t n = 0; n < _accums.size(); ++n)
    {
	if (_nframes > 0)
	{
	    double	tmp = _accums[n].tv_sec * 1.0e6 + _accums[n].tv_usec;
	    out << setw(9) << tmp / (1.0e3 * _nframes) << "ms("
		<< setw(7) << 1.0e6 * _nframes / tmp << "fps)";
	}
	else
	    out << setw(9) << '*' << "ms(" << setw(7) << '*' << "fps)";

	total.tv_sec  += _accums[n].tv_sec;
	total.tv_usec += _accums[n].tv_usec;
    }
    double	tmp = total.tv_sec * 1.0e6 + total.tv_usec;
    return out << '|' << setw(8) << tmp / (1.0e3 * _nframes)
	       << "ms(" << setw(7) << 1.0e6 * _nframes / tmp
	       << "fps)" << endl;
}

std::ostream&
Profiler<true>::printTabSeparated(std::ostream& out) const
{
    timeval		total;
    total.tv_sec = total.tv_usec = 0;

    for (size_t n = 0; n < _accums.size(); ++n)
    {
	if (_nframes > 0)
	{
	    double	tmp = _accums[n].tv_sec * 1.0e6 + _accums[n].tv_usec;
	    out << tmp / (1.0e3 * _nframes);
	}

	total.tv_sec  += _accums[n].tv_sec;
	total.tv_usec += _accums[n].tv_usec;
	out << '\t';
    }
    double	tmp = total.tv_sec * 1.0e6 + total.tv_usec;
    return out << "| " << tmp / (1.0e3 * _nframes) << std::endl;
}

}

