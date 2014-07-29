/*
 *  平成21-22年（独）産業技術総合研究所 著作権所有
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
 *  Copyright 2009-2010.
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
#include "TU/FeatureMatch.h"

namespace TU
{
/************************************************************************
*  class FeatureMatch::Sampler						*
************************************************************************/
//! 指定された個数の点対応をランダムに取り出す．
/*!
  \param npoints	取り出すべき点対応の個数
  \return		取り出された点対応の並び
*/
FeatureMatch::Sampler::Container
FeatureMatch::Sampler::sample(size_t npoints) const
{
    using namespace	std;

    if (npoints > _size)
	throw runtime_error("Sampler<C>::sample(): not enough point matches!");
    
    Container	samples;
    while (samples.size() != npoints)
    {
	const_iterator	item = _begin;
	advance(item, _random.generateInt32() % _size);
	if (find(samples.begin(), samples.end(), *item) == samples.end())
	    samples.push_back(*item);
    }
    
    return samples;
}

}
