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
 *  $Id: Movie.h,v 1.7 2009-09-11 05:56:13 ueshiba Exp $
 */
#ifndef __TUMovie_h
#define __TUMovie_h

#include <utility>
#include "TU/Image++.h"
#include "TU/Manip.h"

namespace TU
{
/************************************************************************
*  class Movie<T>							*
************************************************************************/
//! ムービーを表すクラス
/*!
  複数のviewを持つことができ，そのサイズは各view毎に別個に指定できる．
  \param T	画素の型
*/
template <class T> class Movie
{
  private:
    class View : public Image<T>
    {
      public:
	View()	:Image<T>(), offset(0)			{}
	
	u_int	offset;
    };
    
  public:
    Movie()								;
    ~Movie()								;

  // Inquire movie status.
			operator bool()				const	;
    u_int		nframes()				const	;
    u_int		nviews()				const	;
    u_int		currentFrame()				const	;
    u_int		currentView()				const	;
    u_int		width()					const	;
    u_int		height()				const	;
    const Image<T>&	image()					const	;
    Image<T>&		image()						;

  // Change movie status.
    Movie&		setFrame(u_int frame)				;
    Movie&		setView(u_int view)				;
    Movie&		rewind()					;
    Movie&		operator ++()					;
    Movie&		operator --()					;

  // Allocate frames.
    u_int		alloc(const Array<std::pair<u_int, u_int> >& sizes,
			      u_int nf)					;

  // Restore/Save movie.
    std::istream&	restore(std::istream& in)			;
    std::ostream&	save(std::ostream& out,
			     ImageBase::Type type=ImageBase::DEFAULT)	;
    ImageBase::Type	saveHeader(std::ostream& out,
				   ImageBase::Type
				       type=ImageBase::DEFAULT)	const	;
    std::ostream&	saveFrame(std::ostream& out,
				  ImageBase::Type
				      type=ImageBase::DEFAULT)	const	;

  private:
    ImageBase::Type	restoreHeader(std::istream& in)			;
    std::istream&	restoreFrames(std::istream& in,
				      ImageBase::Type type, u_int m)	;
    static u_int	nelements(u_int npixels)			;
    
  private:
    Array<View>		_views;
    u_int		_cView;			// current view #.
    u_int		_nelements;		// # of elements per frame.
    Array<Array<T> >	_frames;
    u_int		_cFrame;		// current frame #.
#ifdef _DEBUG
    std::ofstream	_log;
#endif
};

//! ムービーを生成する．
template <class T> inline
Movie<T>::Movie()
    :_views(0), _cView(0), _nelements(0), _frames(0), _cFrame(0)
{
#ifdef _DEBUG
    _log.open("Movie.log");
#endif
}

//! ムービーを破壊する．
template <class T> inline
Movie<T>::~Movie()
{
#ifdef _DEBUG
    _log.close();
#endif
}

//! ムービーの状態が正常であるか調べる．
/*!
  \return	現在のフレームが「末尾の次」に達していればfalse, そうでなければtrue
*/
template <class T> inline
Movie<T>::operator bool() const
{
    return (_cView < nviews() && _cFrame < nframes());
}

//! フレーム数を返す．
/*!
  \return	フレーム数
*/
template <class T> inline u_int
Movie<T>::nframes() const
{
    return _frames.dim();
}

//! view数を返す．
/*!
  \return	view数
*/
template <class T> inline u_int
Movie<T>::nviews() const
{
    return _views.dim();
}

//! 現在のフレーム番号を返す．
/*!
  \return	フレーム番号
*/
template <class T> inline u_int
Movie<T>::currentFrame() const
{
    return _cFrame;
}

//! 現在のview番号を返す．
/*!
  \return	view番号
*/
template <class T> inline u_int
Movie<T>::currentView() const
{
    return _cView;
}

//! 現在のviewの画像の幅を返す．
/*!
  \return	画像の幅
*/
template <class T> inline u_int
Movie<T>::width() const
{
    return (_cView < nviews() ? _views[_cView].width() : 0);
}
    
//! 現在のviewの画像の高さを返す．
/*!
  \return	画像の高さ
*/
template <class T> inline u_int
Movie<T>::height() const
{
    return (_cView < nviews() ? _views[_cView].height() : 0);
}

//! 現在のviewとframeに対応する画像を返す．
/*!
  \return	画像
*/
template <class T> inline const Image<T>&
Movie<T>::image() const
{
    return _views[_cView];
}

//! 現在のviewとframeに対応する画像を返す．
/*!
  \return	画像
*/
template <class T> inline Image<T>&
Movie<T>::image()
{
    return _views[_cView];
}

//! 現在のフレームを指定する．
/*!
  frame < #nframes() でない場合は現在フレームは #nframes() となり，
  #operator ()でfalseが返される状態になる．
  \param frame	フレーム番号
  \return	このムービーオブジェクト
*/
template <class T> Movie<T>&
Movie<T>::setFrame(u_int frame)
{
#ifdef _DEBUG
    using namespace	std;
    _log << "  Begin: Movie<T>::setFrame(frame = " << frame << ")" << endl;
#endif
    if (frame != _cFrame)
    {
	if (frame < nframes())
	{
	    _cFrame = frame;

	    for (u_int i = 0; i < nviews(); ++i)
		_views[i].resize((T*)_frames[_cFrame] + _views[i].offset,
				 _views[i].height(), _views[i].width());
	}
	else
	    _cFrame = nframes();
    }
#ifdef _DEBUG
    _log << "  End:   Movie<T>::setFrame()" << endl;
#endif
    return *this;
}

//! 現在のviewを指定する．
/*!
  view < #nviews() でない場合は何も変更しない．
  \param view	view番号
  \return	このムービーオブジェクト
*/
template <class T> inline Movie<T>&
Movie<T>::setView(u_int view)
{
#ifdef _DEBUG
    using namespace	std;
    _log << "  Begin: Movie<T>::setView(view = " << view << ")" << endl;
#endif
    if (view < nviews())
	_cView = view;
#ifdef _DEBUG
    _log << "  End:   Movie<T>::setView()" << endl;
#endif
    return *this;
}

//! 現在のフレームを最初(0)に戻す．
/*!
  \return	このムービーオブジェクト
*/
template <class T> inline Movie<T>&
Movie<T>::rewind()
{
    return setFrame(0);
}

//! 現在のフレームを1つ先に進める．
/*!
  既に最後のフレームに達している場合はフレーム番号を#nframes()にする．
  \return	このムービーオブジェクト
*/
template <class T> inline Movie<T>&
Movie<T>::operator ++()
{
    if (_cFrame < nframes())
	setFrame(_cFrame + 1);
    return *this;
}

//! 現在のフレームを1つ戻す．
/*!
  既に最初のフレームに達している場合は何もしない．
  \return	このムービーオブジェクト
*/
template <class T> inline Movie<T>&
Movie<T>::operator --()
{
    if (_cFrame > 0)
	setFrame(_cFrame - 1);
    return *this;
}

//! 各viewのサイズとフレーム数を指定してムービーの記憶領域を確保する．
/*!
  確保後は，現在のviewとフレームを共に0に設定する．
  \param sizes	各viewの幅と高さのペアを収めた配列．配列のサイズがview数となる．
  \param nf	フレーム数
  \return	実際に確保されたフレーム数
*/
template <class T> u_int
Movie<T>::alloc(const Array<std::pair<u_int, u_int> >& sizes, u_int nf)
{
#ifdef _DEBUG
    using namespace	std;
    _log << "  Begin: Movie<T>::alloc(nframes = " << nf
	 << ", nviews = " << sizes.dim()
	 << ")" << endl;
#endif
  // 各viewのオフセットと1フレームあたりの画素数を設定．
    _views.resize(sizes.dim());
    _nelements = 0;
    for (u_int i = 0; i < nviews(); ++i)
    {
	_views[i].offset = _nelements;
	_nelements += nelements(sizes[i].first * sizes[i].second);
    }
	     
  // 指定された枚数のフレームを設定．
    _frames.resize(nf);
    for (u_int j = 0; j < _frames.dim(); ++j)
    {
	try
	{
	    _frames[j].resize(_nelements);
#ifdef _DEBUG
	    _log << "    " << j << "-th frame allocated..." << endl;
#endif
	}
	catch (...)
	{
	    _frames.resize(j);
#ifdef _DEBUG
	    _log << "    " << j << "-th frame cannot be allocated!" << endl;
#endif
	    break;
	}
    }
    
  // 指定された個数のviewとその大きさを設定．
    if (nframes() > 0)
	for (u_int i = 0; i < nviews(); ++i)
	    _views[i].resize((T*)_frames[0] + _views[i].offset,
			     sizes[i].second, sizes[i].first);
    
    _cFrame = 0;
    _cView  = 0;
#ifdef _DEBUG
    _log << "  End:   Movie<T>::alloc()" << endl;
#endif
    return nframes();			// Return #frames allocated correctly.
}

//! 入力ストリームからムービーを読み込む．
/*!
  入力ストリームの末尾に達するまでフレームを読み続ける．
  \param in	入力ストリーム
  \return	inで指定した入力ストリーム
*/
template <class T> inline std::istream&
Movie<T>::restore(std::istream& in)
{
    return restoreFrames(in, restoreHeader(in), 0);
}

template <class T> ImageBase::Type
Movie<T>::restoreHeader(std::istream& in)
{
    using namespace	std;
    
    char	c;
    if (!in.get(c))
	return ImageBase::DEFAULT;
    if (c != 'M')
	throw runtime_error("TU::Movie<T>::restoreHeader: not a movie file!!");
    
    u_int	nv;
    in >> nv >> skipl;
    _views.resize(nv);

    ImageBase::Type	type = ImageBase::DEFAULT;
    _nelements = 0;
    for (u_int i = 0; i < nviews(); ++i)
    {
	type = _views[i].restoreHeader(in);
	_views[i].offset = _nelements;
	_nelements += nelements(_views[i].width() * _views[i].height());
    }

    return type;
}

template <class T> std::istream&
Movie<T>::restoreFrames(std::istream& in, ImageBase::Type type, u_int m)
{
    char	c;
    if (!in.get(c))
    {
	_frames.resize(m);
	return in;
    }
    in.putback(c);

    try
    {
	Array<T>	frame(_nelements);
	for (u_int i = 0; i < nviews(); ++i)
	{
	    _views[i].resize((T*)frame + _views[i].offset,
			      _views[i].height(), _views[i].width());
	    _views[i].restoreData(in, type);
	}
	restoreFrames(in, type, m + 1);
	_frames[m] = frame;
    }
    catch (...)
    {
	_frames.resize(m);
    }
        
    return in;
}

//! ムービーを指定した画素タイプで出力ストリームに書き出す．
/*!
 \param out	出力ストリーム
 \param type	画素タイプ．ただし，#DEFAULTを指定した場合は，このムービーの
		画素タイプで書き出される．   
 \return	outで指定した出力ストリーム
*/
template <class T> std::ostream&
Movie<T>::save(std::ostream& out, ImageBase::Type type)
{
    u_int	cFrame = _cFrame;

    saveHeader(out, type);
    for (rewind(); *this; ++(*this))
	saveFrame(out, type);
    setFrame(_cFrame);
    
    return out;
}

//! ムービーのヘッダを指定した画素タイプで出力ストリームに書き出す．
/*!
 \param out	出力ストリーム
 \param type	画素タイプ．ただし，#DEFAULTを指定した場合は，このムービーの
		画素タイプで書き出される．   
 \return	実際に書き出す場合の画素タイプ
*/
template <class T> ImageBase::Type
Movie<T>::saveHeader(std::ostream& out, ImageBase::Type type) const
{
    using namespace	std;
    
    out << 'M' << nviews() << endl;
    for (u_int i = 0; i < nviews(); ++i)
	type = _views[i].saveHeader(out, type);
    return type;
}

//! 現在のフレームを指定した画素タイプで出力ストリームに書き出す．
/*!
 \param out	出力ストリーム
 \param type	画素タイプ．ただし，#DEFAULTを指定した場合は，このムービーの
		画素タイプで書き出される．   
 \return	outで指定した出力ストリーム
*/
template <class T> std::ostream&
Movie<T>::saveFrame(std::ostream& out, ImageBase::Type type) const
{
    for (u_int i = 0; i < nviews(); ++i)
	_views[i].saveData(out, type);
    return out;
}

template <class T> inline u_int
Movie<T>::nelements(u_int npixels)
{
    return npixels;
}

template <> inline u_int
Movie<YUV411>::nelements(u_int npixels)
{
    return npixels / 2;
}

}
#endif	// !__TUMovie_h
