/*
 *  $Id: Widget-Xaw.h,v 1.2 2002-07-25 02:38:13 ueshiba Exp $
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
