/*!
 *   \file	CmdPane.h
 */
#ifndef TU_CMDPANE_H
#define TU_CMDPANE_H

#include <QWidget>
#include <QGridLayout>
#include <QPushButton>

namespace TU
{
/************************************************************************
*  class CmdPane							*
************************************************************************/
class CmdPane : public QWidget
{
    Q_OBJECT

  public:
		CmdPane(QWidget* parent)				;

    template <class CAMERA>
    void	addCmds(CAMERA& camera)					;

  Q_SIGNALS:
    void	timerSet(bool enable)					;

  private:
    template <class CAMERA>
    void	addFormatAndFeatureCmds(CAMERA& camera)			;

  private:
    QGridLayout* const	_layout;
};

inline
CmdPane::CmdPane(QWidget* parent)
    :QWidget(parent),
     _layout(new QGridLayout(this))
{
    _layout->setHorizontalSpacing(2);
    _layout->setVerticalSpacing(2);
}

template <class CAMERA> void
CmdPane::addCmds(CAMERA& camera)
{
  // カメラからの画像取り込みをon/offするtoggle buttonを生成．
    const auto	toggle = new QPushButton(tr("Capture"), this);
    toggle->setCheckable(true);
    connect(toggle, &QPushButton::toggled,
	    [&camera, this](bool enable)
	    {
		camera.continuousShot(enable);
		this->timerSet(enable);
	    });
    toggle->setChecked(camera.inContinuousShot());
    _layout->addWidget(toggle, 0, 0, 1, 1);

    addFormatAndFeatureCmds(camera);
}
    
}	// namespace TU
#endif	// !TU_CMDPANE_H
