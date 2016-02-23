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
 *  $Id: mmInstructions.h 1826 2015-05-07 01:47:56Z ueshiba $
 */
/*!
  \file		BufTraits.h
  \brief	CPUのSIMD命令に関連するクラスと関数の定義と実装
*/
#if !defined(__TU_SIMD_BUFTRAITS_H)
#define __TU_SIMD_BUFTRAITS_H

#include <algorithm>
#include "TU/simd/store_iterator.h"
#include "TU/simd/load_iterator.h"

namespace TU
{
template <class T, class ALLOC>	struct BufTraits;
template <class T, class ALLOC>
struct BufTraits<simd::vec<T>, ALLOC>
{
    typedef simd::allocator<simd::vec<T> >	allocator_type;
    typedef typename allocator_type::pointer	pointer;
    typedef simd::store_iterator<T*, true>	iterator;
    typedef simd::load_iterator<const T*, true>	const_iterator;

    static pointer	alloc(allocator_type& allocator, size_t siz)
			{
			    pointer	p = allocator.allocate(siz);
			    for (pointer q = p, qe = q + siz; q != qe; ++q)
				allocator.construct(q, T());
			    return p;
			}
    
    static void		free(allocator_type& allocator, pointer p, size_t siz)
			{
			    for (pointer q = p, qe = q + siz; q != qe; ++q)
				allocator.destroy(q);
			    allocator.deallocate(p, siz);
			}

    template <class IN_, class OUT_>
    static OUT_		copy(IN_ ib, IN_ ie, OUT_ out)
			{
			    return std::copy(ib, ie, out);
			}

    template <class ITER_, class T_>
    static void		fill(ITER_ ib, ITER_ ie, const T_& c)
			{
			    std::fill(ib, ie, c);
			}
};
}	// namespace TU
#endif	// !__TU_SIMD_BUFTRAITS_H
