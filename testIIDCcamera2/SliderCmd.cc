/*!
 *  \file	SliderCmd.cc
 */
#include <QBoxLayout>
#include <cmath>
#include "SliderCmd.h"

namespace TU
{
/************************************************************************
*  class SliderCmd							*
************************************************************************/
SliderCmd::SliderCmd(QWidget* parent)
    :QWidget(parent),
     _slider(new QSlider(Qt::Horizontal, this)),
     _spinBox(new QDoubleSpinBox(this))
{
    const auto	layout = new QBoxLayout(QBoxLayout::LeftToRight, this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(_slider);
    layout->addWidget(_spinBox);

    connect(_slider,  &QAbstractSlider::valueChanged,
	    this,     &SliderCmd::onSliderValueChanged);
    connect(_spinBox, static_cast<sig_p>(&QDoubleSpinBox::valueChanged),
	    this,     &SliderCmd::onSpinBoxValueChanged);
}

double
SliderCmd::value() const
{
    return _spinBox->value();
}

void
SliderCmd::setValue(double val)
{
    _spinBox->setValue(val);
}

void
SliderCmd::setRange(double min, double max, double step)
{
  // setRange() がsignalを発生する可能性があるので，一時的にslotを遮断
    disconnect(_spinBox, static_cast<sig_p>(&QDoubleSpinBox::valueChanged),
	       this,     &SliderCmd::onSpinBoxValueChanged);

    _slider->setSingleStep(1);
    _spinBox->setRange(min, max);

    if (min == std::floor(min) &&
	max == std::floor(max) && step == std::floor(step))
    {
	if (step == 0)
	    step = 1;

	_slider ->setRange(0, (int(max) - int(min))/int(step));
	_spinBox->setSingleStep(step);
	_spinBox->setDecimals(0);
    }
    else
    {
	if (step == 0)
	    step = (max - min) / 1000;

	_slider ->setRange(0, 1000);
	_spinBox->setSingleStep(step);
	const auto	decimal = std::max(0, 3 + int(-std::log10(max-min)));
	_spinBox->setDecimals(decimal);
    }

  // slotを再接続
    connect(_spinBox, static_cast<sig_p>(&QDoubleSpinBox::valueChanged),
	    this,     &SliderCmd::onSpinBoxValueChanged);
}

void
SliderCmd::onSliderValueChanged(int sliderVal)
{
    if (sliderVal != valToSliderVal(_spinBox->value()))
    {
	const auto	val = sliderValToVal(sliderVal);
	_spinBox->setValue(val);
	valueChanged(val);			// signalを発行
    }
}

void
SliderCmd::onSpinBoxValueChanged(double val)
{
    _slider->setValue(valToSliderVal(val));
    valueChanged(val);				// signalを発行
}

double
SliderCmd::sliderValToVal(int sliderVal) const
{
    const auto  max  = _spinBox->maximum();
    const auto  min  = _spinBox->minimum();
    const auto  step = _spinBox->singleStep();

    if (min == std::floor(min) &&
	max == std::floor(max) && step == std::floor(step))
    {
	return min + sliderVal*step;
    }
    else
    {
	return (step > 0 ?
		min + int(sliderVal*(max - min)/(1000*step))*step :
		min + sliderVal*(max - min)/1000);
    }
}

int
SliderCmd::valToSliderVal(double val) const
{
    const auto  max  = _spinBox->maximum();
    const auto  min  = _spinBox->minimum();
    const auto  step = _spinBox->singleStep();

    if (min == std::floor(min) &&
	max == std::floor(max) && step == std::floor(step))
    {
	return (int(val) - int(min))/int(step);
    }
    else
    {
	return int(1000*(val - min)/(max - min));
    }
}

}	// namespace TU
