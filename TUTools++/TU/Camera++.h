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
 *  $Id: Camera++.h 1937 2016-02-20 02:46:39Z ueshiba $
 */
/*!
  \file		Camera++.h
  \brief	カメラ内部パラメータおよびカメラ自身に関するクラスの定義と実装
*/
#ifndef __TU_CAMERAPP_H
#define __TU_CAMERAPP_H

#include "TU/Geometry++.h"

namespace TU
{
/************************************************************************
*  class IntrinsicBase<T>						*
************************************************************************/
//! 全成分が固定されたカメラ内部パラメータを表すクラス
template <class T>
class IntrinsicBase
{
  public:
    using element_type	= T;			//!< 数値計算全般のベースとなる型
    using point2_type	= Point2<element_type>;		//!< 2次元点の型
    using vector_type	= Vector<element_type>;		//!< ベクトルの型
    using matrix_type	= Matrix<element_type>;		//!< 行列の型
    using matrix22_type	= Matrix<element_type, 2, 2>;	//!< 2x2行列の型
    using matrix33_type	= Matrix<element_type, 3, 3>;	//!< 3x3行列の型

  public:
    IntrinsicBase(element_type k=1, const point2_type& u0={0, 0},
		  element_type a=1, element_type s=0)			;
    IntrinsicBase(const matrix33_type& K)				;
    
    point2_type		u(const point2_type& x)			const	;
    point2_type		xFromU(const point2_type& u)		const	;
    matrix22_type	jacobian(const point2_type& x)		const	;

    element_type	k()					const	;
    const point2_type&	u0()					const	;
    element_type	aspect()				const	;
    element_type	skew()					const	;
    element_type	d1()					const	;
    element_type	d2()					const	;
    matrix33_type	K()					const	;
    matrix33_type	Kt()					const	;
    matrix33_type	Kinv()					const	;
    matrix33_type	Ktinv()					const	;
    void		setFocalLength(element_type k)			;
    void		setPrincipal(const point2_type& u0)		;
    void		setAspect(element_type a)			;
    void		setSkew(element_type s)				;
    void		setK(const matrix33_type& K)			;
    void		setDistortion(element_type, element_type)	;
    
    constexpr static size_t
			dofIntrinsic()					;
    void		updateIntrinsic(const vector_type& dp)		;

    std::istream&	get(std::istream& in)				;
    std::ostream&	put(std::ostream& out)			const	;

  protected:
    element_type	k00()					const	;
    element_type	k01()					const	;
    point2_type		u(const point2_type& x,
			  matrix_type* J,matrix_type* H)	const	;

  protected:
    element_type	_k;		//!< 焦点距離
    point2_type		_u0;		//!< 画像主点
    element_type	_k00;		//!< 焦点距離とアスペクト比の積
    element_type	_k01;		//!< 焦点距離と非直交性歪みの積
};
    
//! 内部パラメータをセットして初期化する．
/*!
  \param k	焦点距離
  \param u0	画像主点
  \param a	アスペクト比
  \param s	非直交性歪み
*/
template <class T> inline
IntrinsicBase<T>::IntrinsicBase(element_type k, const point2_type& u0,
				element_type a, element_type s)
    :_k(k), _u0(u0), _k00(a*_k), _k01(s*_k)
{
}

//! 内部パラメータをセットして初期化する．
/*!
  \param K	3x3内部パラメータ行列
*/
template <class T> inline
IntrinsicBase<T>::IntrinsicBase(const matrix33_type& K)
    :_k(1), _u0(0, 0), _k00(1), _k01(0)
{
    setK(K);
}

//! canonical画像座標系において表現された投影点の画像座標系における位置を求める．
/*!
  \param x	canonical画像座標における投影点の2次元位置
  \return	xの画像座標系における2次元位置，すなわち
		\f$
		\TUvec{u}{} =
		\TUbeginarray{cc} ak & sk \\ & k \TUendarray \TUvec{x}{}
		+ \TUvec{u}{0}
		\f$
*/
template <class T> inline typename IntrinsicBase<T>::point2_type
IntrinsicBase<T>::u(const point2_type& x) const
{
    return {_k00*x[0] + _k01*x[1] + _u0[0], _k*x[1] + _u0[1]};
}
    
//! 画像座標における投影点の2次元位置をcanonical画像座標系に直す．
/*!
  \param u	画像座標系における投影点の2次元位置
  \return	canonical画像座標系におけるuの2次元位置，すなわち
		\f$
		\TUvec{x}{} =
		\TUbeginarray{cc}
		a^{-1}k^{-1} & -a^{-1}sk^{-1} \\ & k^{-1}
		\TUendarray
		(\TUvec{u}{} - \TUvec{u}{0})
		\f$
*/
template <class T> inline typename IntrinsicBase<T>::point2_type
IntrinsicBase<T>::xFromU(const point2_type& u) const
{
    return {(u[0] - _u0[0] - (u[1] - _u0[1])*_k01/_k)/_k00,
	    (u[1] - _u0[1])/_k};
}
    
//! canonical画像座標に対する画像座標の1階微分を求める．
/*!
  \param x	canonical画像座標における投影点の2次元位置
  \return	canonical画像座標に対する画像座標の1階微分，すなわち
		\f$
		\TUvec{J}{x} =
		\TUbeginarray{cc}
		ak & sk \\ & k
		\TUendarray
		\f$
*/
template <class T> inline typename IntrinsicBase<T>::matrix22_type
IntrinsicBase<T>::jacobian(const point2_type& x) const
{
    return {{_k00, _k01}, {0, _k}};
}
    
//! 焦点距離を返す．
/*!
  \return	焦点距離
*/
template <class T> inline typename IntrinsicBase<T>::element_type
IntrinsicBase<T>::k() const
{
    return _k;
}
    
//! 画像主点を返す．
/*!
  \return	画像主点
*/
template <class T>
inline const typename IntrinsicBase<T>::point2_type&
IntrinsicBase<T>::u0() const
{
    return _u0;
}

//! アスペクト比を返す．
/*!
  \return	アスペクト比
*/
template <class T> inline typename IntrinsicBase<T>::element_type
IntrinsicBase<T>::aspect() const
{
    return _k00 / _k;
}
    
//! 非直交性歪みを返す．
/*!
  \return	非直交性歪み
*/
template <class T> inline typename IntrinsicBase<T>::element_type
IntrinsicBase<T>::skew() const
{
    return _k01 / _k;
}
    
//! 放射歪曲の第1係数を返す．
/*!
  \return	放射歪曲の第1係数
*/
template <class T> inline typename IntrinsicBase<T>::element_type
IntrinsicBase<T>::d1() const
{
    return 0;
}
    
//! 放射歪曲の第2係数を返す．
/*!
  \return	放射歪曲の第2係数
*/
template <class T> inline typename IntrinsicBase<T>::element_type
IntrinsicBase<T>::d2() const
{
    return 0;
}
    
//! 内部パラメータ行列を返す．
/*!
  \return	3x3内部パラメータ行列，すなわち
		\f$
		\TUvec{K}{} =
		\TUbeginarray{ccc}
		ak & sk & u_0 \\ & k & v_0 \\ & & 1
		\TUendarray
		\f$
*/
template <class T> inline typename IntrinsicBase<T>::matrix33_type
IntrinsicBase<T>::K() const
{
    return {{_k00, _k01, _u0[0]}, {0, _k, _u0[1]}, {0, 0, 1}};
}

//! 内部パラメータ行列の転置を返す．
/*!
  \return	3x3内部パラメータ行列の転置，すなわち
		\f$
		\TUtvec{K}{} =
		\TUbeginarray{ccc}
		ak & & \\ sk & k & \\ u_0 & v_0 & 1
		\TUendarray
		\f$
*/
template <class T> inline typename IntrinsicBase<T>::matrix33_type
IntrinsicBase<T>::Kt() const
{
    return {{_k00, 0, 0}, {_k01, _k, 0}, {_u0[0], _u0[1], 1}};
}

//! 内部パラメータ行列の逆行列を返す．
/*!
  \return	3x3内部パラメータ行列の逆行列，すなわち
		\f$
		\TUinv{K}{} =
		\TUbeginarray{ccc}
		a^{-1}k^{-1} & -a^{-1}k^{-1}s &
		-a^{-1}k^{-1}(u_0 - s v_0) \\ & k^{-1} & -k^{-1}v_0 \\ & & 1
		\TUendarray
		\f$
*/
template <class T> inline typename IntrinsicBase<T>::matrix33_type
IntrinsicBase<T>::Kinv() const
{
    return {{1/_k00, -_k01/(_k00*_k), -_u0[0]/_k00 + (_u0[1]*_k01)/(_k00*_k)},
	    {0,	      1/_k,	      -_u0[1]/_k			    },
	    {0,	      0,	       1				    }};
}

//! 内部パラメータ行列の転置の逆行列を返す．
/*!
  \return	3x3内部パラメータ行列の転置の逆行列，すなわち
		\f$
		\TUtinv{K}{} =
		\TUbeginarray{ccc}
		a^{-1}k^{-1} & & \\ -a^{-1}k^{-1}s & k^{-1} & \\
		-a^{-1}k^{-1}(u_0 - s v_0) & -k^{-1}v_0 & 1
		\TUendarray
		\f$
*/
template <class T> inline typename IntrinsicBase<T>::matrix33_type
IntrinsicBase<T>::Ktinv() const
{
    return {{1/_k00,				      0,	 0},
	    {-_k01/(_k00*_k),			      1/_k,	 0},
	    {-_u0[0]/_k00 + (_u0[1]*_k01)/(_k00*_k), -_u0[1]/_k, 1}};
}

//! 焦点距離を設定する．
/*!
  \param k	焦点距離
*/
template <class T> inline void
IntrinsicBase<T>::setFocalLength(element_type k)
{
    _k00 *= (k / _k);
    _k01 *= (k / _k);
    _k = k;
}

//! 画像主点を設定する．
/*!
  \param u0	画像主点
*/
template <class T> inline void
IntrinsicBase<T>::setPrincipal(const point2_type& u0)
{
    _u0 = u0;
}

//! アスペクト比を設定する．
/*!
  \param a	アスペクト比
*/
template <class T> inline void
IntrinsicBase<T>::setAspect(element_type a)
{
    _k00 = a * _k;
}
    
//! 非直交性歪みを設定する．
/*!
  \param s	非直交性歪み
*/
template <class T> inline void
IntrinsicBase<T>::setSkew(element_type s)
{
    _k01 = s * _k;
}
    
//! 放射歪曲係数以外の内部パラメータを設定する．
/*!
  \param K	3x3内部パラメータ行列
*/
template <class T> inline void
IntrinsicBase<T>::setK(const matrix33_type& K)
{
    setFocalLength(K[1][1]/K[2][2]);
    setPrincipal({K[0][2]/K[2][2], K[1][2]/K[2][2]});
    setAspect(K[0][0] / K[1][1]);
    setSkew(K[0][1] / K[1][1]);
}

//! 放射歪曲係数を設定する．
/*!
 *  実際には何もしないダミー関数
 */
template <class T> inline void
IntrinsicBase<T>::setDistortion(element_type, element_type)
{
}
    
//! 可変内部パラメータの自由度を返す．
/*!
  \return	可変内部パラメータの自由度, すなわち0
*/
template <class T> inline constexpr size_t
IntrinsicBase<T>::dofIntrinsic()
{
    return 0;
}
    
//! 内部パラメータを指定された量だけ更新する．
/*!
  \param dp	更新量を表す dofIntrinsic() 次元ベクトル
*/
template <class T> inline void
IntrinsicBase<T>::updateIntrinsic(const vector_type&)
{
}
    
//! 入力ストリームからカメラの内部パラメータを読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
template <class T> inline std::istream&
IntrinsicBase<T>::get(std::istream& in)
{
    in >> _k >> _u0 >> _k00 >> _k01;
    _k00 *= _k;
    _k01 *= _k;
    return in;
}

//! 出力ストリームにカメラの内部パラメータを書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
template <class T> inline std::ostream&
IntrinsicBase<T>::put(std::ostream& out) const
{
    std::cerr << "Focal length:    ";
    out << _k << std::endl;
    std::cerr << "Principal point:";
    out << _u0;
    std::cerr << "Aspect ratio:    ";
    out << aspect() << std::endl;
    std::cerr << "Skew:            ";
    return out << skew() << std::endl;
}

//! 焦点距離とアスペクト比の積を返す．
/*!
  \return	焦点距離とアスペクト比の積
*/
template <class T> inline typename IntrinsicBase<T>::element_type
IntrinsicBase<T>::k00() const
{
    return _k00;
}
    
//! 焦点距離と非直交性歪みの積を返す．
/*!
  \return	焦点距離と非直交性歪みの積
*/
template <class T> inline typename IntrinsicBase<T>::element_type
IntrinsicBase<T>::k01() const
{
    return _k01;
}

template <class T> typename IntrinsicBase<T>::point2_type
IntrinsicBase<T>::u(const point2_type& x, matrix_type* J, matrix_type* H) const
{
    if (J)
    {
      // [外部パラメータに関する1階微分]
      //   (放射歪曲を付加された)canonical画像座標系から画像座標系に変換
	slice<6>((*J)[0], 0)  = _k00 * slice<6>((*J)[0], 0)
			      + _k01 * slice<6>((*J)[1], 0);
	slice<6>((*J)[1], 0) *= _k;

	if (H)
	{
	  // [外部パラメータに関する2階微分]
	  //   (放射歪曲を付加された)canonical画像座標系から画像座標系に変換
	    slice<6, 6>(H[0], 0, 0)  = _k00 * slice<6, 6>(H[0], 0, 0)
				     + _k01 * slice<6, 6>(H[1], 0, 0);
	    slice<6, 6>(H[1], 0, 0) *= _k;

	    symmetrize(H[0]);
	    symmetrize(H[1]);
	}
    }

    return u(x);
}

/************************************************************************
*  class IntrinsicWithFocalLength<T>					*
************************************************************************/
//! 焦点距離のみが可変なカメラ内部パラメータを表すクラス
template <class T>
class IntrinsicWithFocalLength : public IntrinsicBase<T>
{
  private:
    using	super = IntrinsicBase<T>;

  public:
    using	typename super::element_type;
    using	typename super::point2_type;
    using	typename super::vector_type;
    using	typename super::matrix_type;
    
  public:
    IntrinsicWithFocalLength(element_type k=1,
			     const point2_type& u0={0, 0},
			     element_type a=1, element_type s=0)	;

    using	super::u;
    using	super::jacobian;
    
    constexpr static size_t
		dofIntrinsic()						;
    void	updateIntrinsic(const vector_type& dp)			;

  protected:
    point2_type	u(const point2_type& x,
		  matrix_type* J, matrix_type* H)		const	;
};

//! 内部パラメータをセットして初期化する．
/*!
  \param k	焦点距離
  \param u0	画像主点
  \param a	アスペクト比
  \param s	非直交性歪み
*/
template <class T> inline
IntrinsicWithFocalLength<T>::IntrinsicWithFocalLength(
    element_type k, const point2_type& u0, element_type a, element_type s)
    :super(k, u0, a, s)
{
}

//! 可変内部パラメータの自由度を返す．
/*!
  \return	可変内部パラメータの自由度, すなわち1
*/
template <class T> inline constexpr size_t
IntrinsicWithFocalLength<T>::dofIntrinsic()
{
    return 1;
}
    
//! 内部パラメータを指定された量だけ更新する．
/*!
  \param dp	更新量を表す dofIntrinsic() 次元ベクトル
*/
template <class T> inline void
IntrinsicWithFocalLength<T>::updateIntrinsic(const vector_type& dp)
{
    const auto	a = super::aspect();
    const auto	s = super::skew();
    super::_k -= dp[0];
    setAspect(a);
    setSkew(s);
}
    
template <class T> typename IntrinsicWithFocalLength<T>::point2_type
IntrinsicWithFocalLength<T>::u(const point2_type& x,
			       matrix_type* J, matrix_type* H) const
{
    if (J)
    {
      // 焦点距離に関する1階微分
	(*J)[0][6] = super::aspect() * x[0] + super::skew() * x[1];
	(*J)[1][6] =					      x[1];

	if (H)
	{
	  // 焦点距離と外部パラメータに関する2階微分
	    slice<6>(H[0][6], 0) = super::aspect() * slice<6>((*J)[0], 0)
				 + super::skew()   * slice<6>((*J)[1], 0);
	    slice<6>(H[1][6], 0) =		     slice<6>((*J)[1], 0);
	}
    }
    
    return super::u(x, J, H);
}
    
/************************************************************************
*  class IntrinsicWithEuclideanImagePlane<T>				*
************************************************************************/
//! 焦点距離と画像主点のみが可変なカメラ内部パラメータを表すクラス
template <class T>
class IntrinsicWithEuclideanImagePlane : public IntrinsicWithFocalLength<T>
{
  private:
    using	super = IntrinsicBase<T>;

  public:
    using	typename super::element_type;
    using	typename super::point2_type;
    using	typename super::vector_type;
    using	typename super::matrix_type;
    
  public:
    IntrinsicWithEuclideanImagePlane(element_type k=1,
				     const point2_type& u0={0, 0},
				     element_type a=1, element_type s=0);

    using	super::u;
    using	super::jacobian;
    
    constexpr static size_t
		dofIntrinsic()						;
    void	updateIntrinsic(const vector_type& dp)			;

  protected:
    point2_type	u(const point2_type& x,
		  matrix_type* J, matrix_type* H)		const	;
};

//! 内部パラメータをセットして初期化する．
/*!
  \param k	焦点距離
  \param u0	画像主点
  \param a	アスペクト比
  \param s	非直交性歪み
*/
template <class T> inline
IntrinsicWithEuclideanImagePlane<T>::IntrinsicWithEuclideanImagePlane(
    element_type k, const point2_type& u0, element_type a, element_type s)
    :super(k, u0, a, s)
{
}

//! 可変内部パラメータの自由度を返す．
/*!
  \return	可変内部パラメータの自由度, すなわち3
*/
template <class T> inline constexpr size_t
IntrinsicWithEuclideanImagePlane<T>::dofIntrinsic()
{
    return super::dofIntrinsic() + 2;
}
    
//! 内部パラメータを指定された量だけ更新する．
/*!
  \param dp	更新量を表す dofIntrinsic() 次元ベクトル
*/
template <class T> inline void
IntrinsicWithEuclideanImagePlane<T>::updateIntrinsic(const vector_type& dp)
{
    constexpr auto	j = super::dofIntrinsic();
    super::updateIntrinsic(dp(0, j));
    super::_u0[0] -= dp[j];
    super::_u0[1] -= dp[j + 1];
}
    
template <class T> typename IntrinsicWithEuclideanImagePlane<T>::point2_type
IntrinsicWithEuclideanImagePlane<T>::u(const point2_type& x,
				       matrix_type* J, matrix_type* H) const
{
    if (J)
    {
      // [画像主点に関する1階微分]
	constexpr auto	j = 6 + super::dofIntrinsic();
	(*J)[0][j] = (*J)[1][j+1] = 1;
    }
    
    return super::u(x, J, H);
}
    
/************************************************************************
*  class Intrinsic<T>							*
************************************************************************/
//! 放射歪曲係数以外の全パラメータが可変なカメラ内部パラメータを表すクラス
template <class T>
class Intrinsic : public IntrinsicBase<T>
{
  private:
    using	super = IntrinsicBase<T>;

  public:
    using	typename super::element_type;
    using	typename super::point2_type;
    using	typename super::vector_type;
    using	typename super::matrix_type;
    
  public:
    Intrinsic(element_type k=1, const point2_type& u0={0, 0},
	      element_type a=1, element_type s=0)			;

    constexpr static size_t
		dofIntrinsic()						;
    void	updateIntrinsic(const vector_type& dp)			;

    using	super::u;
    using	super::jacobian;
    
  protected:
    point2_type	u(const point2_type& x,
		  matrix_type* J, matrix_type* H)		const	;
};

//! 内部パラメータをセットして初期化する．
/*!
  \param k	焦点距離
  \param u0	画像主点
  \param a	アスペクト比
  \param s	非直交性歪み
*/
template <class T> inline
Intrinsic<T>::Intrinsic(element_type k, const point2_type& u0,
			element_type a, element_type s)
    :super(k, u0, a, s)
{
}

//! 可変内部パラメータの自由度を返す．
/*!
  \return	可変内部パラメータの自由度, すなわち5
*/
template <class T> inline constexpr size_t
Intrinsic<T>::dofIntrinsic()
{
    return 5;
}
    
//! 内部パラメータを指定された量だけ更新する．
/*!
  \param dp	更新量を表す dofIntrinsic() 次元ベクトル
*/
template <class T> inline void
Intrinsic<T>::updateIntrinsic(const vector_type& dp)
{
    super::_k     -= dp[0];
    super::_u0[0] -= dp[1];
    super::_u0[1] -= dp[2];
    super::_k00   -= dp[3];
    super::_k01   -= dp[4];
}
    
template <class T> typename Intrinsic<T>::point2_type
Intrinsic<T>::u(const point2_type& x, matrix_type* J, matrix_type* H) const
{
    if (J)
    {
      // [内部パラメータに関する1階微分]
	(*J)[0][7]  = (*J)[1][8] = 1;
	(*J)[0][9]		 = x[0];
	(*J)[0][10] = (*J)[1][6] = x[1];

	if (H)
	{
	  // [内部パラメータと外部パラメータに関する2階微分]
	    slice<2, 6>(H[0], 9, 0) = slice<2, 6>(*J, 0, 0);
	    slice<6>(H[1][6], 0)    = slice<6>((*J)[1], 0);
	}
    }
    
    return super::u(x, J, H);
}
    
/************************************************************************
*  class IntrinsicWithDistortion<I>					*
************************************************************************/
//! 放射歪曲係数を含む全パラメータが可変なカメラ内部パラメータを表すクラス
template <class I>
class IntrinsicWithDistortion : public I
{
  private:
    using	super = I;

  public:
    using	typename super::element_type;
    using	typename super::point2_type;
    using	typename super::vector_type;
    using	typename super::matrix_type;
    using	typename super::matrix22_type;
    using	typename super::matrix33_type;
    
  public:
    IntrinsicWithDistortion(element_type k=1,
			    const point2_type& u0={0, 0},
			    element_type a=1,  element_type s=0,
			    element_type d1=0, element_type d2=0)	;

    point2_type		u(const point2_type& x)			const	;
    point2_type		xd(const point2_type& x)		const	;
    point2_type		xFromU(const point2_type& u)		const	;
    matrix22_type	jacobian(const point2_type& x)		const	;
    element_type	d1()					const	;
    element_type	d2()					const	;
    void		setDistortion(element_type d1, element_type d2)	;
    
    constexpr static size_t
			dofIntrinsic()					;
    void		updateIntrinsic(const vector_type& dp)		;

    std::istream&	get(std::istream& in)				;
    std::ostream&	put(std::ostream& out)			const	;
    
  protected:
    point2_type		u(const point2_type& x,
			  matrix_type* J, matrix_type* H)	const	;

  private:
    vector_type		DDu(const point2_type& x)		const	;
    
  private:
    element_type	_d1;
    element_type	_d2;
};

//! 内部パラメータをセットして初期化する．
/*!
  \param k	焦点距離
  \param u0	画像主点
  \param a	アスペクト比
  \param s	非直交性歪み
  \param d1	放射歪曲の第1係数
  \param d2	放射歪曲の第2係数
*/
template <class I> inline
IntrinsicWithDistortion<I>::IntrinsicWithDistortion(
    element_type k, const point2_type& u0, element_type a, element_type s,
    element_type d1, element_type d2)
    :super(k, u0, a, s), _d1(d1), _d2(d2)
{
}
    
//! canonical画像座標系において表現された投影点の画像座標系における位置を求める．
/*!
  \param x	canonical画像座標における投影点の2次元位置
  \return	xの画像座標系における2次元位置，すなわち
		\f$
		\TUvec{u}{} =
		\TUbeginarray{cc} ak & sk \\ & k \TUendarray \TUbreve{x}{}
		+ \TUvec{u}{0},~~
		\TUbreve{x}{} = (1 + d_1 r^2 + d_2 r^4)\TUvec{x}{},~~
		r = \TUnorm{\TUvec{x}{}}
		\f$
*/
template <class I> inline typename IntrinsicWithDistortion<I>::point2_type
IntrinsicWithDistortion<I>::u(const point2_type& x) const
{
    return super::u(xd(x));
}
    
//! canonical座標系において表現された投影点に放射歪曲を付加する．
/*!
  \param x	canonical画像座標における投影点の2次元位置
  \return	放射歪曲付加後のcanonical座標系における位置，すなわち
		\f$
		\TUbreve{x}{} = (1 + d_1 r^2 + d_2 r^4)\TUvec{x}{},~~
		r = \TUnorm{\TUvec{x}{}}
		\f$
*/
template <class I> inline typename IntrinsicWithDistortion<I>::point2_type
IntrinsicWithDistortion<I>::xd(const point2_type& x) const
{
    const auto	sqr = x * x;
    const auto	tmp = 1 + sqr*(_d1 + sqr*_d2);
    return {tmp * x[0], tmp * x[1]};
}
    
//! 画像座標における投影点の2次元位置をcanonical画像座標系に直す．
/*!
  \param u	画像座標系における投影点の2次元位置
  \return	canonical画像カメラ座標系におけるuの2次元位置，すなわち
		\f$
		\TUvec{x}{} =
		(1 - d_1 \breve{r}^2 - d_2 \breve{r}^4)\TUbreve{x}{},~~
		\breve{r} = \TUnorm{\TUbreve{x}{}}
		\f$
*/
template <class I> inline typename IntrinsicWithDistortion<I>::point2_type
IntrinsicWithDistortion<I>::xFromU(const point2_type& u) const
{
    const auto	xd  = super::xFromU(u);
    const auto	sqr = xd * xd;
    const auto	tmp = 1 - sqr*(_d1 + sqr*_d2);
    return {tmp*xd[0], tmp*xd[1]};
}
    
//! canonical画像座標に対する画像座標の1階微分を求める．
/*!
  \param x	canonical画像座標における投影点の2次元位置
  \return	canonical画像座標に対する画像座標の1階微分，すなわち
		\f$
		\TUvec{J}{x} =
		\TUbeginarray{cc}
		ak & sk \\ & k
		\TUendarray
		\f$
*/
template <class I> inline typename IntrinsicWithDistortion<I>::matrix22_type
IntrinsicWithDistortion<I>::jacobian(const point2_type& x) const
{
    const auto	sqr = x * x;
    const auto	tmp = 1 + sqr*(_d1 + sqr*_d2);
    const auto	J   = diag<2>(tmp) + ((2*_d1 + 4*sqr*_d2) * x) % x;
    
    return super::jacobian(xd(x)) * J;
}
    
//! 放射歪曲の第1係数を返す．
/*!
  \return	放射歪曲の第1係数
*/
template <class I> inline typename IntrinsicWithDistortion<I>::element_type
IntrinsicWithDistortion<I>::d1() const
{
    return _d1;
}
    
//! 放射歪曲の第2係数を返す．
/*!
  \return	放射歪曲の第2係数
*/
template <class I> inline typename IntrinsicWithDistortion<I>::element_type
IntrinsicWithDistortion<I>::d2() const
{
    return _d2;
}
    
//! 放射歪曲係数を設定する．
/*!
  \param d1	放射歪曲の第1係数
  \param d2	放射歪曲の第2係数
*/
template <class I> inline void
IntrinsicWithDistortion<I>::setDistortion(element_type d1, element_type d2)
{
    _d1 = d1;
    _d2 = d2;
}
    
//! 可変内部パラメータの自由度を返す．
/*!
  \return	可変内部パラメータの自由度
*/
template <class I> inline constexpr size_t
IntrinsicWithDistortion<I>::dofIntrinsic()
{
    return super::dofIntrinsic() + 2;
}
    
//! 内部パラメータを指定された量だけ更新する．
/*!
  \param dp	更新量を表す dofIntrinsic() 次元ベクトル
*/
template <class I> inline void
IntrinsicWithDistortion<I>::updateIntrinsic(const vector_type& dp)
{
    constexpr auto	j = super::dofIntrinsic();
    super::updateIntrinsic(dp(0, j));
    _d1 -= dp[j];
    _d2 -= dp[j + 1];
}

//! 入力ストリームからカメラの内部パラメータを読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
template <class I> inline std::istream&
IntrinsicWithDistortion<I>::get(std::istream& in)
{
    super::get(in);
    return in >> _d1 >> _d2;
}

//! 出力ストリームにカメラの内部パラメータを書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
template <class I> inline std::ostream&
IntrinsicWithDistortion<I>::put(std::ostream& out) const
{
    super::put(out);
    std::cerr << "Distortion-1:    ";
    out << _d1 << std::endl;
    std::cerr << "Distortion-2:    ";
    return out << _d2 << std::endl;
}

template <class I> typename IntrinsicWithDistortion<I>::point2_type
IntrinsicWithDistortion<I>::u(const point2_type& x,
			      matrix_type* J, matrix_type* H) const
{
    if (J)
    {
      // [放射歪曲パラメータに関する1階微分]
	const auto	sqr = x * x;
	constexpr auto	i = 6 + super::dofIntrinsic();
	(*J)[0][i]   = sqr * (super::k00() * x[0] + super::k01() * x[1]);
	(*J)[1][i]   = sqr * (			    super::k()   * x[1]);
	(*J)[0][i+1] = sqr * (*J)[0][i];
	(*J)[1][i+1] = sqr * (*J)[1][i];

	const matrix22_type	Dxd = diag<2>(1 + sqr*(_d1 + sqr*_d2))
				    + ((2*_d1 + 4*sqr*_d2) * x) % x;

	if (H)
	{
	  // [外部パラメータに関する2階微分]
	  //   canonical座標系から放射歪曲を付加されたcanonical座標系に変換
	    const matrix_type	tmp = Dxd[1][0] * slice<6, 6>(H[0], 0, 0);
	    slice<6, 6>(H[0], 0, 0) = Dxd[0][0] * slice<6, 6>(H[0], 0, 0)
				    + Dxd[0][1] * slice<6, 6>(H[1], 0, 0);
	    slice<6, 6>(H[1], 0, 0) = Dxd[1][1] * slice<6, 6>(H[1], 0, 0)
				    + tmp;

	  // [放射歪曲と外部パラメータに関する2階微分]
	    const matrix22_type
		DDxd({{ 2*x[0]*x[0] + sqr,	2*x[0]*x[1]},
		      {(4*x[0]*x[0] + sqr)*sqr,	4*x[0]*x[1]*sqr}});
	    const matrix22_type
		DDyd({{2*x[0]*x[1],	 2*x[1]*x[1] + sqr},
		      {4*x[0]*x[1]*sqr, (4*x[1]*x[1] + sqr)*sqr}});
	    slice<2, 6>(H[0], i, 0) = (super::k00()*DDxd + super::k01()*DDyd)
				    * slice<2, 6>(*J, 0, 0);
	    slice<2, 6>(H[1], i, 0) = (super::k()*DDyd) * slice<2, 6>(*J, 0, 0);

	  // [放射歪曲と内部パラメータに関する2階微分]
	    constexpr auto	DOF = super::dofIntrinsic();
	    slice<DOF>(H[0][i  ], 6) = sqr * DDu(x);
	    slice<DOF>(H[0][i+1], 6) = sqr * slice<DOF>(H[0][i], 6);
	    H[1][i  ][6] = sqr * x[1];
	    H[1][i+1][6] = sqr * H[1][i][6];
	}

      // [外部パラメータに関する1階微分]
      //   canonical座標系から放射歪曲を付加されたcanonical座標系に変換
	slice<2, 6>(*J, 0, 0) = Dxd * slice<2, 6>(*J, 0, 0);
    }
    
    return super::u(xd(x), J, H);
}

template <class I> inline typename IntrinsicWithDistortion<I>::vector_type
IntrinsicWithDistortion<I>::DDu(const point2_type& x) const
{
    vector_type	val(super::dofIntrinsic());
    val[0] = super::aspect() * x[0] + super::skew() * x[1];

    return val;
}

template <> inline IntrinsicWithDistortion<Intrinsic<float> >::vector_type
IntrinsicWithDistortion<Intrinsic<float> >::DDu(const point2_type& x) const
{
    vector_type	val(super::dofIntrinsic());
    val[3] = x[0];
    val[4] = x[1];

    return val;
}

template <> inline IntrinsicWithDistortion<Intrinsic<double> >::vector_type
IntrinsicWithDistortion<Intrinsic<double> >::DDu(const point2_type& x) const
{
    vector_type	val(super::dofIntrinsic());
    val[3] = x[0];
    val[4] = x[1];

    return val;
}

/************************************************************************
*  class CanonicalCamera<T>						*
************************************************************************/
//! すべての内部パラメータが標準既定値(焦点距離とアスペクト比が1, 非直交性歪みと放射歪曲係数が0, 画像主点が原点に一致)となる透視投影カメラを表すクラス
template <class T>
class CanonicalCamera
{
  public:
  //! 数値計算全般のベースとなる型
    using element_type	= T;
  //! 2次元点の型
    using point2_type	= Point2<element_type>;
  //! 3次元点の型
    using point3_type	= Point3<element_type>;
  //! 2次元直線の型
    using line_type	= HyperPlane<Vector<element_type, 3> >;
  //! 3次元平面の型
    using plane_type	= HyperPlane<Vector<element_type, 4> >;
  //! ベクトルの型
    using vector_type	= Vector<element_type>;
  //! 行列の型
    using matrix_type	= Matrix<element_type>;
  //! 2x3行列の型
    using matrix23_type	= Matrix<element_type, 2, 3>;
  //! 3x3行列の型
    using matrix33_type	= Matrix<element_type, 3, 3>;
  //! 3x4行列の型
    using matrix34_type	= Matrix<element_type, 3, 4>;
  //! 4x4行列の型
    using matrix44_type	= Matrix<element_type, 4, 4>;

  public:
    CanonicalCamera(const point3_type& t={0, 0, 0},
		    const matrix33_type&
			Rt={{1, 0, 0}, {0, 1, 0}, {0, 0, 1}})		;

    point2_type		x(const point3_type& X, matrix_type* J=nullptr,
			  matrix_type* H=nullptr)		const	;
    matrix23_type	jacobian(const point3_type& X)		const	;
    matrix34_type	Pc()					const	;
    const point3_type&	t()					const	;
    const matrix33_type&
			Rt()					const	;
    void		setTranslation(const point3_type& t)		;
    void		setRotation(const matrix33_type& Rt)		;
    void		move(const matrix44_type& D)			;

    plane_type		h(const line_type& l)			const	;
    point3_type		X(const point2_type& x,
			  const plane_type& h)			const	;
    
    constexpr static size_t
			dof()						;
    void		update(const vector_type& dp)			;
    void		updateFCC(const vector_type& dp)		;

    std::istream&	get(std::istream& in)				;
    std::ostream&	put(std::ostream& out)			const	;

  private:
    point3_type		_t;			//!< カメラの投影中心
    matrix33_type	_Rt;			//!< カメラの回転
};

//! 位置と姿勢をセットして初期化する．
/*!
  \param t	カメラ位置を表す3次元ベクトル
  \param Rt	カメラ姿勢を表す3x3回転行列
*/
template <class T> inline
CanonicalCamera<T>::CanonicalCamera(const point3_type& t,
				    const matrix33_type& Rt)
    :_t(t), _Rt(Rt)
{
}

//! 3次元空間中の点の投影点のcanonical画像座標系における位置を求める．
/*!
  \param X	対象点の3次元位置
  \param J	0でなければカメラの位置・姿勢に関する投影点の1階微分，
		すなわち2x6ヤコビ行列
		\f$
		\TUbeginarray{cc}
		\TUdisppartial{\TUvec{x}{}}{\TUvec{t}{}} &
		\TUdisppartial{\TUvec{x}{}}{\TUvec{\theta}{}}
		\TUendarray
		\f$
		が返される
  \param H	0でなければカメラの位置・姿勢に関する投影点の2階微分，
		すなわち2つの6x6ヘッセ行列
		\f$
		\TUbeginarray{cc}
		\frac{\partial^2 x}{\partial\TUvec{t}{}^2} &
		\frac{\partial^2 x}
		     {\partial\TUvec{t}{}\partial\TUvec{\theta}{}} \\
		\frac{\partial^2 x}
		     {\partial\TUvec{\theta}{}\partial\TUvec{t}{}} &
		\frac{\partial^2 x}{\partial\TUvec{\theta}{}^2}
		\TUendarray
		\f$, 
		\f$
		\TUbeginarray{cc}
		\frac{\partial^2 y}{\partial\TUvec{t}{}^2} &
		\frac{\partial^2 y}
		     {\partial\TUvec{t}{}\partial\TUvec{\theta}{}} \\
		\frac{\partial^2 y}
		     {\partial\TUvec{\theta}{}\partial\TUvec{t}{}} &
		\frac{\partial^2 y}{\partial\TUvec{\theta}{}^2}
		\TUendarray
		\f$
		が返される
  \return	canonical画像座標系におけるxの投影点の位置，すなわち
		\f$
		\TUvec{x}{} = 
		\frac{1}{\TUtvec{r}{z}(\TUvec{X}{} - \TUvec{t}{})}
		\TUbeginarray{c}
		\TUtvec{r}{x}(\TUvec{X}{} - \TUvec{t}{}) \\
		\TUtvec{r}{y}(\TUvec{X}{} - \TUvec{t}{})
		\TUendarray
		\f$
*/
template <class T> typename CanonicalCamera<T>::point2_type
CanonicalCamera<T>::x(const point3_type& X,
		      matrix_type* J, matrix_type* H) const
{
    const point3_type	dX = X - _t;
    const point3_type	Xc = _Rt * dX;
    const point2_type	x({Xc[0]/Xc[2], Xc[1]/Xc[2]});

    if (J)	// ヤコビ行列を計算するなら...
    {
	if (J->nrow() < 2 || J->ncol() < 6)
	    J->resize(2, 6);				// ヤコビ行列は2x6
	
	for (int n = 0; n < 2; ++n)
	{
	    auto	Dx = slice<3>((*J)[n], 0);

	    Dx = (_Rt[2] * x[n] - _Rt[n]) / Xc[2];	// 位置に関する微分
	    slice<3>((*J)[n], 3) = Dx ^ dX;		// 回転に関する微分
	
	    if (H)	// ヘッセ行列を計算するなら...
	    {
		auto&	HH = H[n];			// 第n軸のヘッセ行列
		if (HH.nrow() < 6 || HH.ncol() < 6)
		    HH.resize(6, 6);			// ヘッセ行列は6x6

		slice<3, 3>(HH, 0, 0) = (_Rt[2] % Dx + Dx % _Rt[2]) / Xc[2];
		slice<3, 3>(HH, 0, 3) = skew(Dx) + slice<3, 3>(HH, 0, 0) ^ dX;
		slice<3, 3>(HH, 3, 0) = transpose(slice<3, 3>(HH, 0, 3));
		slice<3, 3>(HH, 3, 3) = slice<3, 3>(HH, 0, 3) ^ dX;
		(slice<3, 3>(HH, 3, 3) += transpose(slice<3, 3>(HH, 3, 3)))
				       *= element_type(0.5);
	    }
	}
    }
    
    return x;
}

template <class T> inline typename CanonicalCamera<T>::matrix23_type
CanonicalCamera<T>::jacobian(const point3_type& X) const
{
    const point3_type	Xc = _Rt*(X - _t);
    const point2_type	x{{Xc[0]/Xc[2], Xc[1]/Xc[2]}};
    matrix23_type	J;
    J[0] = (_Rt[0] - x[0]*_Rt[2])/Xc[2];
    J[1] = (_Rt[1] - x[1]*_Rt[2])/Xc[2];
    
    return J;
}

//! 3次元ユークリッド空間からcanonical画像平面への投影行列を求める．
/*!
  \return	canonical画像平面への投影行列，すなわち
		\f$
		\TUvec{P}{c} = \TUtvec{R}{}
		\TUbeginarray{cc}
		\TUvec{I}{3\times 3} & -\TUvec{t}{}
		\TUendarray
		\f$
*/
template <class T> inline typename CanonicalCamera<T>::matrix34_type
CanonicalCamera<T>::Pc() const
{
    matrix34_type	P;
    P(0, 3, 0, 3) = _Rt;
    P[0][3] = -(_Rt[0] * _t);
    P[1][3] = -(_Rt[1] * _t);
    P[2][3] = -(_Rt[2] * _t);

    return P;
}

//! カメラの位置を返す．
/*!
  \return	カメラの3次元位置
*/
template <class T> inline const typename CanonicalCamera<T>::point3_type&
CanonicalCamera<T>::t() const
{
    return _t;
}

//! カメラの姿勢を返す．
/*!
  \return	カメラの姿勢を表す3x3回転行列
*/
template <class T> inline const typename CanonicalCamera<T>::matrix33_type&
CanonicalCamera<T>::Rt() const
{
    return _Rt;
}
    
//! カメラの位置を設定する．
/*!
  \param t	カメラの3次元位置
*/
template <class T> inline void
CanonicalCamera<T>::setTranslation(const point3_type& t)
{
    _t = t;
}

//! カメラの姿勢を設定する．
/*!
  \param Rt	カメラの姿勢を表す3x3回転行列
*/
template <class T> inline void
CanonicalCamera<T>::setRotation(const matrix33_type& Rt)
{
    _Rt = Rt;
}

//! カメラを移動する．
/*!
  \param D	カメラの移動を表す4x4剛体変換行列，すなわち
		\f$
		\TUvec{D}{} =
		\TUbeginarray{cc} \TUtvec{Q}{} & - \TUtvec{Q}{}\TUvec{d}{} \\
		\TUtvec{0}{} & 1
		\TUendarray
		\f$
		ここで\f$\TUvec{d}{}\f$はワールド座標系から見たカメラの移動量，
		\f$\TUvec{Q}{}\f$の各列はワールド座標系から見た移動後の各座標
		軸の向き
*/
template <class T> inline void
CanonicalCamera<T>::move(const matrix44_type& D)
{
    _t[0] += D[0][3];
    _t[1] += D[1][3];
    _t[2] += D[2][3];
    _Rt = evalueate(_Rt * D(0, 3, 0, 3));
}

//! 画像平面上の直線とカメラの投影中心を通る3次元空間中の平面を返す．
/*!
  \param l	画像平面上の直線の同次座標
  \return	lとカメラの投影中心を通る平面の同次座標
*/
template <class T> inline typename CanonicalCamera<T>::plane_type
CanonicalCamera<T>::h(const line_type& l) const
{
    plane_type	h;
    h(0, 3) = l * _Rt;
    h[3] = -(h(0, 3) * _t);

    return h;
}

//! 画像平面上の点を3次元空間中の平面に逆投影した点を返す．
/*!
  \param x	画像平面上の点の非同次座標
  \param h	3次元空間中の平面の同次座標
  \return	xをhに逆投影した3次元空間中の点の非同次座標
*/
template <class T> inline typename CanonicalCamera<T>::point3_type
CanonicalCamera<T>::X(const point2_type& x, const plane_type& h) const
{
    const point3_type	Rx = x[0]*_Rt[0] + x[1]*_Rt[1] + _Rt[2];
    const auto		k  = (h(0, 3)*_t + h[3])/(h(0, 3)*Rx);

    return _t - k*Rx;
}

//! カメラ外部パラメータの自由度を返す．
/*!
  \return	外部パラメータの自由度, すなわち6
*/
template <class T> inline constexpr size_t
CanonicalCamera<T>::dof()
{
    return 6;
}
    
//! カメラの外部パラメータを指定された量だけ更新する．
/*!
  \f$\Delta\TUvec{p}{} =
  [\Delta\TUtvec{t}{},~\Delta\TUtvec{\theta}{}]^\top\f$に対して
  \f{eqnarray*}
  \TUvec{t}{} & \leftarrow & \TUvec{t}{} - \Delta\TUvec{t}{} \\
  \TUtvec{R}{} & \leftarrow &
  \TUtvec{R}{}\TUtvec{R}{}(\Delta\TUvec{\theta}{})
  \f}
  と更新する．
  \param dp	更新量を表す6次元ベクトル
*/
template <class T> inline void
CanonicalCamera<T>::update(const vector_type& dp)
{
    _t -= dp(0, 3);
    updateFCC(dp(3, dp.size() - 3));
}
    
//! カメラの姿勢を指定された量だけ更新する．
/*!
  \f$\Delta\TUvec{p}{} = \Delta\TUvec{\theta}{}\f$に対して
  \f[
  \TUtvec{R}{} \leftarrow \TUtvec{R}{}\TUtvec{R}{}(\Delta\TUvec{\theta}{})
  \f]
  と更新する．カメラの位置は更新されない．
  \param dp	更新量を表す3次元ベクトル
*/
template <class T> inline void
CanonicalCamera<T>::updateFCC(const vector_type& dp)
{
    _Rt = evaluate(_Rt * rotation(dp(0, 3)));
}
    
//! 入力ストリームからカメラの外部パラメータを読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
template <class T> inline std::istream&
CanonicalCamera<T>::get(std::istream& in)
{
    using vector3_type = Vector<element_type, 3>;
    
    constexpr auto	RAD = element_type(M_PI) / element_type(180);
    vector3_type	axis;
    in >> _t >> axis;
    _Rt = rotation(RAD*axis);

    return in;
}

//! 出力ストリームにカメラの外部パラメータを書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
template <class T> inline std::ostream&
CanonicalCamera<T>::put(std::ostream& out) const
{
    constexpr auto	DEG = element_type(180) / element_type(M_PI);
    std::cerr << "Position:       ";
    out << _t;
    std::cerr << "Rotation(deg.): ";
    out << evaluate(DEG * rotation_axis(_Rt));

    return out;
}

//! 入力ストリームからカメラの外部パラメータを読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \param camera	外部パラメータの読み込み先
  \return	inで指定した入力ストリーム
*/
template <class T> inline std::istream&
operator >>(std::istream& in, CanonicalCamera<T>& camera)
{
    return camera.get(in);
}

//! 出力ストリームにカメラの外部パラメータを書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \param camera	外部パラメータの書き出し元
  \return	outで指定した出力ストリーム
*/
template <class T> inline std::ostream&
operator <<(std::ostream& out, const CanonicalCamera<T>& camera)
{
    return camera.put(out);
}

/************************************************************************
*  class Camera<I>							*
************************************************************************/
//! 透視投影カメラを表すクラス
template <class I>
class Camera : public CanonicalCamera<typename I::element_type>, public I
{
  public:
  //! 外部パラメータの型
    using extrinsic_type	= CanonicalCamera<typename I::element_type>;
    using element_type		= typename extrinsic_type::element_type;
    using point2_type		= typename extrinsic_type::point2_type;
    using point3_type		= typename extrinsic_type::point3_type;
    using line_type		= typename extrinsic_type::line_type;
    using plane_type		= typename extrinsic_type::plane_type;
    using vector_type		= typename extrinsic_type::vector_type;
    using matrix_type		= typename extrinsic_type::matrix_type;
    using matrix23_type		= typename extrinsic_type::matrix23_type;
    using matrix33_type		= typename extrinsic_type::matrix33_type;
    using matrix34_type		= typename extrinsic_type::matrix34_type;
    using matrix44_type		= typename extrinsic_type::matrix44_type;
  //! 内部パラメータの型
    using intrinsic_type	= I;
    
  public:
    Camera(const point3_type& t={0, 0, 0},
	   const matrix33_type& Rt={{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
	   const intrinsic_type& intrinsic=intrinsic_type())		;
    Camera(const extrinsic_type& extrinsic,
	   const intrinsic_type& intrinsic=intrinsic_type())		;
    Camera(const matrix34_type& P)					;

    point2_type		operator ()(const point3_type& X,
				    matrix_type* J=nullptr,
				    matrix_type* H=nullptr)	const	;
    matrix23_type	jacobian(const point3_type& X)		const	;
    matrix34_type	P()					const	;
    void		setProjection(const matrix34_type& P)		;

    plane_type		h(const line_type& l)			const	;
    point3_type		X(const point2_type& u,
			  const plane_type& h)			const	;
    
    constexpr static size_t
			dof()						;
    void		update(const vector_type& dp)			;
    void		updateFCC(const vector_type& dp)		;

    std::istream&	get(std::istream& in)				;
    std::ostream&	put(std::ostream& out)			const	;
};

//! 位置と姿勢および内部パラメータをセットして初期化する．
/*!
  \param t		カメラ位置を表す3次元ベクトル
  \param Rt		カメラ姿勢を表す3x3回転行列
  \param intrinsic	内部パラメータ
*/
template <class I> inline
Camera<I>::Camera(const point3_type& t, const matrix33_type& Rt,
		  const intrinsic_type& intrinsic)
    :extrinsic_type(t, Rt), intrinsic_type(intrinsic)
{
}

//! 外部パラメータと内部パラメータをセットして初期化する．
/*!
  \param extrinsic	外部パラメータ
  \param intrinsic	内部パラメータ
*/
template <class I> inline
Camera<I>::Camera(const extrinsic_type& extrinsic,
		  const intrinsic_type& intrinsic)
    :extrinsic_type(extrinsic), intrinsic_type(intrinsic)
{
}

//! 投影行列をセットして初期化する．
/*!
  \param P	3x4投影行列
*/
template <class I> inline
Camera<I>::Camera(const matrix34_type& P)
    :extrinsic_type(), intrinsic_type()
{
    setProjection(P);
}

//! 3次元空間中の点の投影点の画像座標系における位置を求める．
/*!
  \param X	対象点の3次元位置
  \param J	0でなければカメラの位置・姿勢および内部パラメターに関する
		投影点の1階微分，
		すなわち2 x dof() ヤコビ行列
		\f$
		\TUbeginarray{ccc}
		\TUdisppartial{\TUvec{u}{}}{\TUvec{t}{}} &
		\TUdisppartial{\TUvec{u}{}}{\TUvec{\theta}{}} &
		\TUdisppartial{\TUvec{u}{}}{\TUvec{\kappa}{}}
		\TUendarray
		\f$
		が返される
  \param H	0でなければカメラの位置・姿勢に関する投影点の2階微分，
		すなわち2つの dof() x dof() ヘッセ行列
		\f$
		\TUbeginarray{ccc}
		\frac{\partial^2 u}{\partial\TUvec{t}{}^2} &
		\frac{\partial^2 u}
		     {\partial\TUvec{t}{}\partial\TUvec{\theta}{}} &
		\frac{\partial^2 u}
		     {\partial\TUvec{t}{}\partial\TUvec{\kappa}{}} \\
		\frac{\partial^2 u}
		     {\partial\TUvec{\theta}{}\partial\TUvec{t}{}} &
		\frac{\partial^2 u}{\partial\TUvec{\theta}{}^2} &
		\frac{\partial^2 u}
		     {\partial\TUvec{\theta}{}\partial\TUvec{\kappa}{}} \\
		\frac{\partial^2 u}
		     {\partial\TUvec{\kappa}{}\partial\TUvec{t}{}} &
		\frac{\partial^2 u}
		     {\partial\TUvec{\kappa}{}\partial\TUvec{\theta}{}} &
		\TUvec{O}{}
		\TUendarray
		\f$, 
		\f$
		\TUbeginarray{ccc}
		\frac{\partial^2 v}{\partial\TUvec{t}{}^2} &
		\frac{\partial^2 v}
		     {\partial\TUvec{t}{}\partial\TUvec{\theta}{}} &
		\frac{\partial^2 v}
		     {\partial\TUvec{t}{}\partial\TUvec{\kappa}{}} \\
		\frac{\partial^2 v}
		     {\partial\TUvec{\theta}{}\partial\TUvec{t}{}} &
		\frac{\partial^2 v}{\partial\TUvec{\theta}{}^2} &
		\frac{\partial^2 v}
		     {\partial\TUvec{\theta}{}\partial\TUvec{\kappa}{}} \\
		\frac{\partial^2 v}
		     {\partial\TUvec{\kappa}{}\partial\TUvec{t}{}} &
		\frac{\partial^2 v}
		     {\partial\TUvec{\kappa}{}\partial\TUvec{\theta}{}} &
		\TUvec{O}{}
		\TUendarray
		\f$
		が返される
  \return	Xの投影点の画像座標系における位置
*/
template <class I> typename Camera<I>::point2_type
Camera<I>::operator ()(const point3_type& X,
		       matrix_type* J, matrix_type* H) const
{
    if (J)
    {
	J->resize(2, dof());
	
	if (H)
	{
	    H[0].resize(dof(), dof());
	    H[1].resize(dof(), dof());
	}
    }
    
    return intrinsic_type::u(extrinsic_type::x(X, J, H), J, H);
}

template <class I> inline typename Camera<I>::matrix23_type
Camera<I>::jacobian(const point3_type& X) const
{
    return intrinsic_type::jacobian(extrinsic_type::x(X))
	 * extrinsic_type::jacobian(X);
}

//! 3次元ユークリッド空間から画像平面への投影行列を求める．
/*!
  \return	画像平面への投影行列，すなわち
		\f$
		\TUvec{P}{} = \TUvec{K}{}\TUtvec{R}{}
		\TUbeginarray{cc}
		\TUvec{I}{3\times 3} & -\TUvec{t}{}
		\TUendarray
		\f$
*/
template <class I> inline typename Camera<I>::matrix34_type
Camera<I>::P() const
{
    return intrinsic_type::K() * extrinsic_type::Pc();
}

//! 投影行列をセットする．
/*!
  \param P	3x4投影行列
  \return	このカメラ
*/
template <class I> void
Camera<I>::setProjection(const matrix34_type& P)
{
    using vector3_type = Vector<element_type, 3>;

    matrix33_type	K;		// camera intrinsic parameters.
    K[0]    = slice(P[2], 0, 3);
    K[1]    = slice(P[1], 0, 3);
    K[2]    = slice(P[0], 0, 3);
    QRDecomposition<element_type>	qr(K);
    K[0][0] =  qr.Rt()[2][2];
    K[0][1] =  qr.Rt()[2][1];
    K[0][2] = -qr.Rt()[2][0];
    K[1][0] =  0;
    K[1][1] =  qr.Rt()[1][1];
    K[1][2] = -qr.Rt()[1][0];
    K[2][0] =  0;
    K[2][1] =  0;
    K[2][2] = -qr.Rt()[0][0];

    matrix33_type	Rt;		// camera rotation.
    Rt[0]   =  qr.Qt()[2];
    Rt[1]   =  qr.Qt()[1];
    Rt[2]   = -qr.Qt()[0];

    vector3_type	t;		// camera translation.
    t[0]    = -P[0][3];
    t[1]    = -P[1][3];
    t[2]    = -P[2][3];

  // Negate sign of P so that K has positive determinant.
    if (K[0][0] * K[1][1] * K[2][2] < element_type(0))
    {
	K *= -element_type(1);
	t *= -element_type(1);
    }
    
    if (K[0][0] < element_type(0))
    {
	K[0][0] *= -element_type(1);
	Rt[0]   *= -element_type(1);
    }
    if (K[1][1] < element_type(0))
    {
	K[0][1] *= -element_type(1);
	K[1][1] *= -element_type(1);
	Rt[1]   *= -element_type(1);
    }
    if (K[2][2] < element_type(0))
    {
	K[0][2] *= -element_type(1);
	K[1][2] *= -element_type(1);
	K[2][2] *= -element_type(1);
	Rt[2]   *= -element_type(1);
    }
    t = (inverse(K) * t) * Rt;

    extrinsic_type::setTranslation(t);
    extrinsic_type::setRotation(Rt);
    intrinsic_type::setK(K);    
}

//! 画像平面上の直線とカメラの投影中心を通る3次元空間中の平面を返す．
/*!
  \param l	画像平面上の直線の同次座標
  \return	lとカメラの投影中心を通る平面の同次座標
*/
template <class T> inline typename Camera<T>::plane_type
Camera<T>::h(const line_type& l) const
{
    return extrinsic_type::h(l * intrinsic_type::K());
}

//! 画像平面上の点を3次元空間中の平面に逆投影した点を返す．
/*!
  \param u	画像平面上の点の非同次座標
  \param h	3次元空間中の平面の同次座標
  \return	xをhに逆投影した3次元空間中の点の非同次座標
*/
template <class T> inline typename Camera<T>::point3_type
Camera<T>::X(const point2_type& u, const plane_type& h) const
{
    return extrinsic_type::X(intrinsic_type::xFromU(u), h);
}

//! カメラの外部／可変内部パラメータの自由度を返す．
/*!
  \return	外部／可変内部パラメータの自由度, すなわち 6 + dofIntrinsic()
*/
template <class I> inline constexpr size_t
Camera<I>::dof()
{
    return extrinsic_type::dof() + intrinsic_type::dofIntrinsic();
}
    
//! カメラの外部／内部パラメータを指定された量だけ更新する．
/*!
  \f$\Delta\TUvec{p}{} = [\Delta\TUtvec{t}{},~\Delta\TUtvec{\theta}{},
  ~\Delta\TUtvec{\kappa}{}]^\top\f$に対して
  \f{eqnarray*}
  \TUvec{t}{} & \leftarrow & \TUvec{t}{} - \Delta\TUvec{t}{} \\
  \TUtvec{R}{} & \leftarrow &
  \TUtvec{R}{}\TUtvec{R}{}(\Delta\TUvec{\theta}{}) \\
  \TUvec{\kappa}{} & \leftarrow & \TUvec{\kappa}{} - \Delta\TUvec{\kappa}{}
  \f}
  と更新する．
  \param dp	更新量を表す6 + dofIntrinsic() 次元ベクトル
*/
template <class I> inline void
Camera<I>::update(const vector_type& dp)
{
    extrinsic_type::update(dp);
    intrinsic_type::updateIntrinsic(dp(6, dp.size() - 6));
}
    
//! カメラの姿勢と内部パラメータを指定された量だけ更新する．
/*!
  \f$\Delta\TUvec{p}{} = [\Delta\TUtvec{\theta}{},
  ~\Delta\TUtvec{\kappa}{}]^\top\f$に対して
  \f{eqnarray*}
  \TUtvec{R}{} & \leftarrow &
  \TUtvec{R}{}\TUtvec{R}{}(\Delta\TUvec{\theta}{}) \\
  \TUvec{\kappa}{} & \leftarrow & \TUvec{\kappa}{} - \Delta\TUvec{\kappa}{}
  \f}
  と更新する．カメラの位置は更新されない．
  \param dp	更新量を表す3 + dofIntrinsic() 次元ベクトル
*/
template <class I> inline void
Camera<I>::updateFCC(const vector_type& dp)
{
    extrinsic_type::updateFCC(dp);
    intrinsic_type::updateIntrinsic(dp(3, dp.size() - 3));
}
    
//! 入力ストリームからカメラの外部/内部パラメータを読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
template <class I> inline std::istream&
Camera<I>::get(std::istream& in)
{
    extrinsic_type::get(in);
    return intrinsic_type::get(in);
}

//! 出力ストリームにカメラの外部/内部パラメータを書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \return	outで指定した出力ストリーム
*/
template <class I> inline std::ostream&
Camera<I>::put(std::ostream& out) const
{
    extrinsic_type::put(out);
    return intrinsic_type::put(out);
}

//! 入力ストリームからカメラの外部／内部パラメータを読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \param camera	外部／内部パラメータの読み込み先
  \return	inで指定した入力ストリーム
*/
template <class I> inline std::istream&
operator >>(std::istream& in, Camera<I>& camera)
{
    return camera.get(in);
}

//! 出力ストリームにカメラの外部／内部パラメータを書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \param camera	外部／内部パラメータの書き出し元
  \return	outで指定した出力ストリーム
*/
template <class I> inline std::ostream&
operator <<(std::ostream& out, const Camera<I>& camera)
{
    return camera.put(out);
}

}
#endif	// !__TU_CAMERAPP_H
