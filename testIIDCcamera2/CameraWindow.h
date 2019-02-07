/*!
 *   \file	CameraWindow.h
 */
#ifndef TU_CAMERAWINDOW_H
#define TU_CAMERAWINDOW_H

#include <QMainWindow>
#include <QStatusBar>
#include <QTimerEvent>
#include <QElapsedTimer>
#include "CmdPane.h"
#include "ImageView.h"

namespace TU
{
/************************************************************************
*  global functions							*
************************************************************************/
template <class CAMERA> QString
cameraName(const CAMERA& camera)					;

/************************************************************************
*  class CameraWindow<CAMERA>						*
************************************************************************/
template <class CAMERA>
class CameraWindow : public QMainWindow
{
  public:
		CameraWindow(QWidget* parent, CAMERA& camera)		;
		~CameraWindow()						;

    void	onTimerSet(bool enable)					;

  protected:
    void	timerEvent(QTimerEvent* event)				;

  private:
    CAMERA&		_camera;
    ImageView*	const	_imageView;
    CmdPane*	const	_cmdPane;
    int			_timerId;

    int			_nframes;
    QElapsedTimer	_elapsedTimer;
};

template <class CAMERA>
CameraWindow<CAMERA>::CameraWindow(QWidget* parent, CAMERA& camera)
    :QMainWindow(parent),
     _camera(camera),
     _imageView(new ImageView(this, _camera.width(), _camera.height())),
     _cmdPane(new CmdPane(this)),
     _timerId(0),
     _nframes(0)
{
    setAttribute(Qt::WA_DeleteOnClose);
    
    _cmdPane->addCmds(_camera);
    connect(_cmdPane, &CmdPane::timerSet, this, &CameraWindow::onTimerSet);

    const auto	central = new QWidget(this);
    const auto	layout  = new QHBoxLayout(central);
    layout->setContentsMargins(0, 0, 0, 0);
  //layout->setSizeConstraint(QLayout::SetMaximumSize);
    layout->addWidget(_imageView);
    layout->addWidget(_cmdPane);
    setCentralWidget(central);

    setWindowTitle(cameraName(_camera));

    _elapsedTimer.start();
}

template <class CAMERA>
CameraWindow<CAMERA>::~CameraWindow()
{
    onTimerSet(false);
    _camera.continuousShot(false);
}

template <class CAMERA> void
CameraWindow<CAMERA>::onTimerSet(bool enable)
{
    if (enable)
    {
	if (_timerId == 0)
	    _timerId = startTimer(2);	// interval = 2ms (500Hz)
    }
    else
    {
	if (_timerId != 0)
	{
	    killTimer(_timerId);
	    _timerId = 0;
	}
    }
}

template <class CAMERA> void
CameraWindow<CAMERA>::timerEvent(QTimerEvent* event)
{
    if (event->timerId() != _timerId)
	return;

    _imageView->captureAndDisplay(_camera);

    if (_nframes++ == 10)
    {
	const auto	elapsed = _elapsedTimer.elapsed()*1e-3;
	statusBar()->showMessage(QString::number(_nframes/elapsed)
				 .append("fps"));
	_nframes = 0;
	_elapsedTimer.restart();
    }
}

}
#endif	// !TU_CAMERAWINDOW_H
