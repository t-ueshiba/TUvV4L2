/*
 * testIIDCcamera: test program controlling an IIDC 1394-based Digital Camera
 * Copyright (C) 2006 VVV Working Group
 *   National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  $Id: MyIIDCCameraArray.h 822 2012-08-27 02:39:41Z takase $
 */
#ifndef __TUMyIIDCCameraArray_h
#define __TUMyIIDCCameraArray_h

#include <gtk/gtk.h>
#include "MyIIDCCamera.h"

namespace TU
{
/************************************************************************
*  class MyIIDCCameraArray                                                   *
************************************************************************/
/*!
  IEEE1394ボードを表すクラス．GTK+ を用いたカメラ一覧表示のための
  canvas (GTK+ の scrolled window widget)，MyIIDCCameraの各インスタンスを
  生成する機能を持つ．
*/
class MyIIDCCameraArray
{
  public:
    using iterator	 = std::vector<MyIIDCCamera>::iterator;
    using const_iterator = std::vector<MyIIDCCamera>::const_iterator;
    
  public:
    MyIIDCCameraArray()							;
    ~MyIIDCCameraArray()						;

    size_t		size()		const	{ return _cameras.size(); }
    iterator		begin()			{ return _cameras.begin(); }
    const_iterator	begin()		const	{ return _cameras.begin(); }
    iterator		end()			{ return _cameras.end(); }
    const_iterator	end()		const	{ return _cameras.end(); }
    
  //! リストの表示領域となるウィジェットを返す
    GtkWidget*          canvas()				const	;

  //! 現在リストで選択されているカメラの画面を表示する
    GtkWidget*		view()						;

  //! リストの全項目を消去してカメラを解放する
    void		clear()						;

  //! ポートを走査して見つかったIIDCカメラをリストに加える
    void		scan()						;

  //! リストで選択中のカメラについて順番を1つ前にずらす
    void		up()						;

  //! リストで選択中のカメラについて順番を1つ後にずらす
    void		down()						;

  //! 設定ファイルの保存画面ウィジェットを保持する
    void		pushFileSelection(GtkWidget* filesel)		;

  //! 設定ファイルの保存画面ウィジェットを取り出す
    GtkWidget*		popFileSelection()				;

  //! 転送速度選択ウィジェットを設定する
    void		setSpeedPreference(GtkWidget* filesel)		;

  //! 転送速度選択ウィジェットを取得する
    GtkWidget*		getSpeedPreference() const			;

  private:
  //! i番目とj番目のカメラを入れ替える
    void		swap(size_t i, size_t j)			;

  //! カメラを追加する
    void		push_back(MyIIDCCamera&& camera)		;

  private:
    std::vector<MyIIDCCamera>	_cameras;
    
    GtkWidget* const		_canvas;		// リストの表示領域
    GtkWidget*			_list;			// IIDCカメラのリスト
    GtkWidget*			_filesel;		// ファイル選択画面
    GtkWidget*			_speedPreference;	// 転送速度選択ウィジェット
};

inline GtkWidget*
MyIIDCCameraArray::canvas() const
{
    return _canvas;
}

}
#endif /* !__TUMyIIDCCameraArray_h	*/
