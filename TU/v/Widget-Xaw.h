/*
 *  $Id$  
 */
class Widget
{
  public:
    Widget(::Widget widget)				;
    Widget(const Widget& parentWidget,
	   const char*	name, const CmdDef& cmd)	;
    ~Widget()						;

			operator ::Widget()	const	{return _widget;}

    size_t		width()				const	;
    size_t		height()			const	;
    Point2<int>		position()			const	;
    u_long		background()			const	;
    Widget&		setWidth(size_t w)			;
    Widget&		setHeight(size_t h)			;
    Widget&		setPosition(const Point2<int>&)		;
    
  private:
    Widget(const Widget&)					;
    Widget&		operator =(const Widget&)		;

    ::Widget		_widget;
};
