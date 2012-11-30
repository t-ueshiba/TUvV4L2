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
  \file		Movie.h
  \brief	クラス TU::Movie の定義と実装
*/
#ifndef __TUMovie_h
#define __TUMovie_h

#include <utility>
#include <list>
#include "TU/Image++.h"
#include "TU/Manip.h"

namespace TU
{
/************************************************************************
*  class Movie<T>							*
************************************************************************/
//! ムービーを表すクラス
/*!
  複数のビューを持つことができ，そのサイズは各ビュー毎に個別に指定できる．
  \param T	画素の型
*/
template <class T> class Movie
{
  public:
  //! 各ビューの幅と高さのペア
    typedef std::pair<u_int, u_int>			Size;

  private:
  //! ビュー
    struct View : public Image<T>
    {
	View()	:Image<T>(), offset(0)					{}
	
	u_int	offset;		//!< フレームの先頭からの画像データ領域のオフセット
    };

  //! フレーム
    class Frame : public Array<T>
    {
      private:
	typedef Array<T>				super;

      public:
	explicit Frame(u_int n=0)	:super(n)	{super::operator =(0);}
    };

    typedef std::list<Frame>				Frames;
    typedef typename Frames::iterator			iterator;

  public:
    Movie(u_int nviews=0)						;

  // Set sizes for each view.
    Movie<T>&		setSizes(const Array<Size>& sizes)		;

  // General information.
    bool		isCircularMode()			const	;
    Movie<T>&		setCircularMode(bool circular)			;
    u_int		nviews()				const	;
    u_int		width(u_int view)			const	;
    u_int		height(u_int view)			const	;
    const Image<T>&	image(u_int view)			const	;
    Image<T>&		image(u_int view)			;

  // Handling frames.
			operator bool()				const	;
    u_int		nframes()				const	;
    u_int		currentFrame()				const	;
    Movie<T>&		setFrame(u_int frame)				;
    Movie<T>&		rewind()					;
    Movie<T>&		operator ++()					;
    Movie<T>&		operator --()					;
    
  // Edit movie.
    Movie<T>&		insert(u_int n)					;
    const Movie<T>&	copy(u_int n)				const	;
    Movie<T>&		cut(u_int n)					;
    u_int		paste()						;
    Movie<T>&		swap()						;
    
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
    Movie<T>&		setFrameToViews()				;
    ImageBase::TypeInfo	restoreHeader(std::istream& in)			;
    std::istream&	restoreFrames(std::istream& in,
				      ImageBase::TypeInfo typeInfo,
				      u_int m)				;
    static u_int	npixels(u_int n)				;
    
  private:
    bool		_circular;	//!< 循環モード/非循環モード
    Array<View>		_views;		//!< ビューの並び
    Frames		_frames;	//!< フレームの並び
    iterator		_dummy;		//!< フレームの末尾を表すダミーフレーム
    iterator		_current;	//!< 現フレーム
    u_int		_cFrame;	//!< 現フレームの番号
    mutable Frames	_buf;		//!< 編集用バッファ
};

//! ビュー数を指定してムービーを生成する．
/*!
  \param nviews		ビュー数
*/
template <class T> inline
Movie<T>::Movie(u_int nviews)
    :_circular(false), _views(nviews), _frames(1),
     _dummy(_frames.begin()), _current(_dummy), _cFrame(0), _buf()
{
}

//! 各ビューのサイズを指定する．
/*!
  指定後は，現在のビューとフレームを共に0に設定する．
  \param sizes	各ビューの幅と高さのペアを収めた配列．配列のサイズがビュー数となる．
  \return	このムービー
*/
template <class T> Movie<T>&
Movie<T>::setSizes(const Array<Size>& sizes)
{
  // ビュー数と各ビューのオフセットを設定．
    _views.resize(sizes.size());
    u_int	n = 0;
    for (u_int i = 0; i < nviews(); ++i)
    {
	_views[i].offset = n;
	n += npixels(sizes[i].first * sizes[i].second);
    }
    
  // 大きさが1フレームあたりの画素数に等しいダミーフレームを確保．
    _frames.clear();			// 全フレームを廃棄
    _frames.push_front(Frame(n));	// ダミーフレームを確保
    _dummy  = _frames.begin();

  // 各ビューにダミーフレームを設定．
    for (u_int i = 0; i < nviews(); ++i)
	_views[i].resize(_dummy->ptr() + _views[i].offset,
			 sizes[i].second, sizes[i].first);

  // 現フレームをダミーフレームに設定．
    _current = _dummy;
    _cFrame  = 0;
    
    return *this;
}

//! 循環モードであるか調べる．
/*!
  \return	循環モードであればtrue, そうでなければfalse
*/
template <class T> inline bool
Movie<T>::isCircularMode() const
{
    return _circular;
}
    
//! 循環/非循環モードを設定する．
/*!
  循環モードに設定する場合は，現フレームがムービーの末尾であれば先頭に設定する．
  \param circular	循環モードであればtrue, そうでなければfalse
  \return		このムービー
*/
template <class T> Movie<T>&
Movie<T>::setCircularMode(bool circular)
{
    _circular = circular;

    if (_circular && _current == _dummy)
	return rewind();
    else
	return *this;
}
    
//! ビュー数を返す．
/*!
  \return	view数
*/
template <class T> inline u_int
Movie<T>::nviews() const
{
    return _views.size();
}

//! 指定されたビューに対応する画像の幅を返す．
/*!
  \param view	ビュー番号
  \return	画像の幅
*/
template <class T> inline u_int
Movie<T>::width(u_int view) const
{
    return _views[view].width();
}
    
//! 指定されたビューに対応する画像の高さを返す．
/*!
  \param view	ビュー番号
  \return	画像の高さ
*/
template <class T> inline u_int
Movie<T>::height(u_int view) const
{
    return _views[view].height();
}

//! 現在のフレームの指定されたビューに対応する画像を返す．
/*!
  \param view	ビュー番号
  \return	画像
*/
template <class T> inline const Image<T>&
Movie<T>::image(u_int view) const
{
    return _views[view];
}

//! 現在のフレームの指定されたビューに対応する画像を返す．
/*!
  \param view	ビュー番号
  \return	画像
*/
template <class T> inline Image<T>&
Movie<T>::image(u_int view)
{
    return _views[view];
}

//! 現フレームの状態を調べる．
/*!
  \return	現フレームが最後のフレームの次に達していればfalse,
		そうでなければtrue
*/
template <class T> inline
Movie<T>::operator bool() const
{
    return (_current != _dummy);
}

//! フレーム数を返す．
/*!
  \return	フレーム数
*/
template <class T> inline u_int
Movie<T>::nframes() const
{
    return _frames.size() - 1;		// ダミーフレームはカウントしない．
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

//! 現フレームを指定する．
/*!
  frame >= nframes() の場合は現フレームは nframes() が返す値に設定され，
  #operator bool() でfalseが返される状態になる．
  \param frame	フレーム番号
  \return	このムービー
*/
template <class T> inline Movie<T>&
Movie<T>::setFrame(u_int frame)
{
    using namespace	std;

    _cFrame  = min(frame, nframes());
    _current = _frames.begin();
    advance(_current, _cFrame);

    return setFrameToViews();
}
    
//! 現フレームをムービーの先頭に戻す．
/*!
  \return	このムービー
*/
template <class T> inline Movie<T>&
Movie<T>::rewind()
{
    return setFrame(0);
}

//! 現フレームを1つ先に進める．
/*!
  現フレームが既に最後のフレームの次に達していたら( #operator bool() で
  falseが返される状態になっていたら)，何もせずにリターンする．
  現フレームが最後のフレームである場合，循環モードでないならばさらに
  最後のフレームの次に進み， #operator bool() でfalseが返される状態になる．
  循環モードならば先頭フレームに移動する．
  \return	このムービー
*/
template <class T> inline Movie<T>&
Movie<T>::operator ++()
{
    if (_current == _dummy)
	return *this;

    ++_current;
    ++_cFrame;
    if (_current == _dummy && _circular)	// 末尾に達し，
    {						// かつ循環モードならば...
	_current = _frames.begin();		// 先頭に移動する．
	_cFrame  = 0;
    }

    return setFrameToViews();
}

//! 現在のフレームを1つ前に戻す．
/*!
  現フレームがムービーの先頭の場合，循環モードでないならばムービーの
  最後のフレームの次に移動し， #operator bool() でfalseが返される状態になる．
  循環モードならば最後のフレームに移動する．
  \return	このムービー
*/
template <class T> inline Movie<T>&
Movie<T>::operator --()
{
    if (_current == _frames.begin())	// ムービーの先頭ならば...
    {
	_current = _dummy;		// 最後のフレームの次に移動する．
	_cFrame  = nframes();
	
	if (_circular)			// さらに循環モードならば...
	{
	    --_current;			// 最後のフレームに戻る．
	    --_cFrame;
	}
    }
    else
    {
	--_current;
	--_cFrame;
    }

    return setFrameToViews();
}

//! 現フレームの直前に指定した枚数のフレームを挿入する．
/*!
  現フレームは挿入した最初のフレームとなる．
  \param n	挿入する枚数
  \return	このムービー
*/
template <class T> Movie<T>&
Movie<T>::insert(u_int n)
{
    iterator	cur = _current;		// 現フレームを記憶する．

  // 挿入後の _current の再設定に備える．
    if (_current == _frames.begin())	// 先頭に挿入するなら...
	_current = _frames.end();	// 一時的に無効化．
    else				// そうでなければ...
	--_current;			// 挿入位置の1つ前．

  // 現フレームの直前に挿入
    _frames.insert(cur, n, Frame(_dummy->size()));

  // _current に挿入された領域の先頭を再設定する．
  // 先頭からの _current のオフセット _cFrame は変化しない．
    if (_current == _frames.end())	// 先頭に挿入した場合は...
	_current = _frames.begin();	// ムービーの先頭．
    else				// そうでなければ...
	++_current;			// 挿入位置の1つ前の次．

    return setFrameToViews();
}
    
//! 現フレームから指定された枚数のフレームを編集用バッファにコピーする．
/*!
  編集用バッファの内容は上書きされる．現フレームは変化しない．
  \param n	コピーされる枚数
  \return	このムービー
 */
template <class T> const Movie<T>&
Movie<T>::copy(u_int n) const
{
  // コピーされる領域の末尾を検出する．
    iterator	tail = _current;
    std::advance(tail, std::min(n, nframes() - _cFrame));

  // [_current, tail) を _buf にコピーする．
    _buf.assign(_current, tail);

    return *this;
}

//! 現フレームから指定されたフレーム数をカットして編集用バッファに移す．
/*!
  編集用バッファの内容は上書きされる．現フレームはカットされた領域の直後となる．
  \param n	カットされるフレーム数
  \return	このムービー
 */
template <class T> Movie<T>&
Movie<T>::cut(u_int n)
{
  // カットされる領域の末尾を検出する．
    n = std::min(n, nframes() - _cFrame);
    iterator	tail = _current;
    std::advance(tail, n);

  // [_current, tail) を _buf に移す．
    _buf.clear();
    _buf.splice(_buf.begin(), _frames, _current, tail);
    
  // _currentをカットされた領域の直後に再設定する．
  // 先頭からの _current のオフセット _cFrame は変化しない．
    _current = tail;

    if (_current == _dummy && _circular)  // 末尾までカットかつ循環モードならば...
	return rewind();		  // 先頭に戻る．
    else				  // そうでなければ...
	return setFrameToViews();	  // _current を各ビューに設定する．
}

//! 現フレームの直前に編集用バッファの内容を挿入する．
/*!
  編集用バッファは空になる．現フレームは挿入された領域の先頭になる．
  \return	挿入されたフレーム数
 */
template <class T> u_int
Movie<T>::paste()
{
    iterator	cur = _current;		// 現フレームを記憶する．
    u_int	n = _buf.size();	// 編集用バッファ中のフレーム数
    
  // 挿入後の _current の再設定に備える．
    if (_current == _frames.begin())	// 先頭に挿入するなら...
	_current = _frames.end();	// 一時的に無効化．
    else				// そうでなければ...
	--_current;			// 挿入位置の1つ前．

  // _bufの内容を現フレームの直前に挿入する．
    _frames.splice(cur, _buf);

  // _current に挿入された領域の先頭を再設定する．
  // 先頭からの _current のオフセット _cFrame は変化しない．
    if (_current == _frames.end())	// 先頭に挿入した場合は...
	_current = _frames.begin();	// ムービーの先頭．
    else				// そうでなければ...
	++_current;			// 挿入位置の1つ前の次．

    setFrameToViews();

    return n;
}

//! 現フレームの前後を交換する．
/*!
  現フレームは交換前のムービーの先頭になる．
  \return	このムービー
 */
template <class T> inline Movie<T>&
Movie<T>::swap()
{
    iterator	tmp = _frames.begin();	// 交換前の先頭を記憶

    _frames.splice(_frames.begin(), _frames, _current, _dummy);

    _current = tmp;			// 交換前の先頭を現フレームとする
    _cFrame  = nframes() - _cFrame;	// 交換後の前半の長さ = 交換前の後半の長さ
    
    return setFrameToViews();
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

//! ムービーを指定した画素タイプで出力ストリームに書き出す．
/*!
 \param out	出力ストリーム
 \param type	画素タイプ．ただし， #TU::ImageBase::DEFAULT を指定した場合は，
		このムービーの画素タイプで書き出される．   
 \return	outで指定した出力ストリーム
*/
template <class T> std::ostream&
Movie<T>::save(std::ostream& out, ImageBase::Type type)
{
    saveHeader(out, type);

    bool	circular = isCircularMode();
    setCircularMode(false);

    for (rewind(); *this; ++(*this))
	saveFrame(out, type);
    rewind();

    setCircularMode(circular);
    
    return out;
}

//! ムービーのヘッダを指定した画素タイプで出力ストリームに書き出す．
/*!
 \param out	出力ストリーム
 \param type	画素タイプ．ただし， #TU::ImageBase::DEFAULT を指定した場合は，
		このムービーの画素タイプで書き出される．   
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
 \param type	画素タイプ．ただし， #TU::ImageBase::DEFAULT を指定した場合は，
		このムービーの画素タイプで書き出される．   
 \return	outで指定した出力ストリーム
*/
template <class T> std::ostream&
Movie<T>::saveFrame(std::ostream& out, ImageBase::Type type) const
{
    for (u_int i = 0; i < nviews(); ++i)
	_views[i].saveData(out, type);
    return out;
}

/*
 *  private member functions
 */
//! 現フレームを個々のビューにセットする．
/*!
  \return	このムービー
*/ 
template <class T> Movie<T>&
Movie<T>::setFrameToViews()
{
    for (u_int i = 0; i < _views.size(); ++i)
	_views[i].resize(_current->ptr() + _views[i].offset,
			 _views[i].height(), _views[i].width());
    return *this;
}
    
template <class T> ImageBase::TypeInfo
Movie<T>::restoreHeader(std::istream& in)
{
    using namespace	std;

  // ファイルの先頭文字が'M'であることを確認する．
    char	c;
    if (!in.get(c))
	return ImageBase::TypeInfo(ImageBase::DEFAULT);
    if (c != 'M')
	throw runtime_error("TU::Movie<T>::restoreHeader: not a movie file!!");

  // ビュー数を読み込み，そのための領域を確保する．
    u_int	nv;
    in >> nv >> skipl;
    _views.resize(nv);

  // 各ビューのヘッダを読み込み，その画像サイズをセットする．
    ImageBase::TypeInfo	typeInfo(ImageBase::DEFAULT);
    Array<Size>		sizes(nviews());
    for (u_int i = 0; i < nviews(); ++i)
    {
	typeInfo = _views[i].restoreHeader(in);
	sizes[i] = make_pair(_views[i].width(), _views[i].height());
    }
    setSizes(sizes);
    
    return typeInfo;
}

template <class T> std::istream&
Movie<T>::restoreFrames(std::istream& in, ImageBase::TypeInfo typeInfo, u_int m)
{
    for (;;)
    {
      // とりあえずダミーフレームに読み込む．
	for (u_int i = 0; i < nviews(); ++i)
	    if (!_views[i].restoreData(in, typeInfo))
		goto finish;

      // コピーしてダミーフレームの直前に挿入
	_frames.insert(_dummy, *_dummy);
    }

  finish:
    rewind();	// 先頭フレームを現フレームとする．
    
    return in;
}

template <class T> inline u_int
Movie<T>::npixels(u_int n)
{
    return n;
}

template <> inline u_int
Movie<YUV411>::npixels(u_int n)
{
    return n / 2;
}

}
#endif	// !__TUMovie_h
