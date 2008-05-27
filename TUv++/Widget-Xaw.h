/*
 *  平成9-19年（独）産業技術総合研究所 著作権所有
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
 *  Copyright 1997-2007.
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
 *  $Id: Widget-Xaw.h,v 1.5 2008-05-27 11:38:26 ueshiba Exp $
 */
class Widget
{
  public:
    Widget(::Widget widget)				;
    Widget(const Widget& parentWidget,
	   const char*	name, const CmdDef& cmd)	;
    ~Widget()						;

			operator ::Widget()	const	{return _widget;}

    u_int		width()				const	;
    u_int		height()			const	;
    Point2<int>		position()			const	;
    u_long		background()			const	;
    Widget&		setWidth(u_int w)			;
    Widget&		setHeight(u_int h)			;
    Widget&		setPosition(const Point2<int>&)		;
    
  private:
    Widget(const Widget&)					;
    Widget&		operator =(const Widget&)		;

    ::Widget		_widget;
};
