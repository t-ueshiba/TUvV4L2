/*
 *  \file	SliderCmd.h
 */
#ifndef TU_SLIDERCMD_H
#define TU_SLIDERCMD_H

#include <QSlider>
#include <QDoubleSpinBox>

namespace TU
{
/************************************************************************
*  class SliderCmd							*
************************************************************************/
class SliderCmd : public QWidget
{
    Q_OBJECT

  private:
    using sig_p	= void (QDoubleSpinBox::*)(double);

  public:
		SliderCmd(QWidget* parent)				;

    double	value()						const	;
    void	setValue(double val)					;
    void	setRange(double min, double max, double step)		;

  Q_SIGNALS:
    void	valueChanged(double val)				;

  private:
    void	onSliderValueChanged(int sliderVal)			;
    void	onSpinBoxValueChanged(double val)			;

  private:
    double	sliderValToVal(int sliderVal)			const	;
    int		valToSliderVal(double val)			const	;

  private:
    QSlider*		const	_slider;
    QDoubleSpinBox*	const	_spinBox;
};

}	// namespace TU
#endif	// !TU_SLIDERCMD_H
