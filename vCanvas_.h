#ifndef vCanvas_h
#define vCanvas_h

/***********************************************************************

  canvas Widget (subclass of Core)

  Copyright (C) 1997  Toshio Ueshiba

 ***********************************************************************/

/* Parameters:

 Name		     Class		RepType		Default Value
 ----		     -----		-------		-------------
 background	     Background		Pixel		XtDefaultBackground
 border		     BorderColor	Pixel		XtDefaultForeground
 borderWidth	     BorderWidth	Dimension	1
 destroyCallback     Callback		Pointer		NULL
 hSpace 	     HSpace		Dimension	4
 height		     Height		Dimension	0
 mappedWhenManaged   MappedWhenManaged	Boolean		True
 vSpace 	     VSpace		Dimension	4
 width		     Width		Dimension	0
 x		     Position		Position	0
 y		     Position		Position	0

*/

#define XtNginitCallback	"ginitCallback"
#define XtCGinitCallback	"GinitCallback"

/* Class record constants */

typedef struct _VCanvasClassRec		*VCanvasWidgetClass;
typedef struct _VCanvasRec		*VCanvasWidget;

extern WidgetClass			vCanvasWidgetClass;

#endif
