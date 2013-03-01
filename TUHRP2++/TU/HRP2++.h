/*
 *  For the communication between HRP2 control RTC's (currently ReachingRTC
 *  and SequencePlayerRTC) service provider and Vision program.
 *    by nkita 100808
 *
 *  Revised to support coninuous monitoring hand poses and integrated into
 *  new library libTUHRP2++.
 *    by t.ueshiba 130301.
 *    
 *  $Id$
 */
#ifndef __TUHRP2PP_H
#define __TUHRP2PP_H

#include <rtm/CorbaNaming.h>
#include <rtm/RTObject.h>
#include <rtm/CorbaConsumer.h>
#include <rtm/idl/BasicDataType.hh>
#include "ReachingService.h"
#include "SequencePlayerService.h"
#include "ForwardKinematicsService.hh"
#include "WalkGeneratorService.h"
#include "misc.h"
#include <pthread.h>
#include <string>
#include <queue>
#include <boost/circular_buffer.hpp>
#include "TU/Vector++.h"

#ifdef HRP3
#  include "bodyinfo_HRP3.h"
#  include "bodyinfo_HRP2toHRP3.h"
#endif

#ifdef HRP2
#  include "bodyinfo_HRP2YH.h"
#endif

#ifdef HRP2DOF7
#  include "bodyinfo_HRP2DOF7.h"
#endif

namespace TU
{
/************************************************************************
*  class HRP2								*
************************************************************************/
class HRP2
{
  public:
    typedef Matrix44d	Pose;
    typedef u_int64_t	Time;

    struct TimedPose : public Pose
    {
	Time	t;
    };
    
  private:
    enum posture_id
    {
	INITPOS, HALFSIT, CLOTHINIT, DESIREDPOS
    };

    enum mask_id
    {
	HANDS, LEFTHAND, RIGHTHAND, DESIREDMASK, EXCEPTHEAD
    };

  //! HRP2の特定の関節の姿勢を常時監視するスレッド
  /*!
   *  姿勢をリングバッファに保存し，クライアントの要求に応じて指定された時刻にもっとも
   *  近い時刻における姿勢を返す．
   */
    class GetRealPoseThread
    {
      public:
	GetRealPoseThread(const HRP2& hrp2,
			  const char* linkName, u_int capacity)		;
	~GetRealPoseThread()						;

	void		run()						;
	bool		operator ()(Time time, TimedPose& D)	const	;
	void		timeSpan(Time& t0, Time& t1)		const	;
		
      private:
	void*		mainLoop()					;
	static void*	threadProc(void* thread)			;
	
	const HRP2&				_hrp2;
	const std::string			_linkName;
	boost::circular_buffer<TimedPose>	_poses;
	bool					_quit;
	mutable pthread_mutex_t			_mutex;
	pthread_t				_thread;
    };

  //! HRP2の引数を持たないコマンドを実行するスレッド
  /*!
   *  HRP2が目標値に到達するまで呼出側に制御を返さないコマンドについて，これを独立した
   *  スレッドで走らせることにより，コマンド実行中にホスト側が別の作業を行えるようにする．
   */
    class ExecuteCommandThread
    {
      public:
	typedef void	(HRP2::* Command)(bool) const;
	
      public:
	ExecuteCommandThread(const HRP2& hrp2)				;
	~ExecuteCommandThread()						;

	void		run()						;
	void		operator ()(Command command)		const	;
	void		wait()					const	;
	bool		isCompleted()				const	;
    
      private:
	void*		mainLoop()					;
	static void*	threadProc(void* thread)			;

	const HRP2&			_hrp2;
	mutable std::queue<Command>	_commands;
	bool				_quit;
	mutable pthread_mutex_t		_mutex;
	mutable pthread_cond_t		_cond;
	pthread_t			_thread;
    };

  public:
    HRP2(int argc, char* argv[],
	 const char* linkName="RARM_JOINT6", u_int capacity=100)	;

    void	setup(bool isLeft, bool isLaterMode)			;
    void	getMotion()						;

    u_int	getMotionLength()				const	;
    bool	getPosture(u_int rank, double*& q, double*& p,
			   double*& rpy, double*& zmp)			;

  // ReachingService
    bool	SelectArm(bool isLeft)				const	;
    bool	DeSelectArm(bool isLeft)			const	;
    bool	isSelected(bool isLeft)				const	;
    bool	isMaster(bool isLeft)				const	;
    bool	SetGraspCenter(bool isLeft, const double* pos)	const	;
    bool	SelectUsedDofs(const bool* used)		const	;
    bool	SelectTaskDofs(bool isLeft,
			       const bool* constrained,
			       const double* weights)		const	;
    bool	SelectExecutionMode(bool isLaterMode)		const	;
    bool	SetTargetPose(bool isLeft, const double* pose,
			      double duration)			const	;
    bool	SetTargetArc(bool isLeft, const double* rot,
			     const double* cen,
			     const double* ax,
			     double theta, double duration)	const	;
    bool	SetTargetVelocity(bool isLeft,
				  const double* velocity,
				  double duration)		const	;
    bool	GetCurrentPose(bool isLeft, double* pose)	const	;
    void	GenerateMotion(bool blocked=true)		const	;
    void	PlayMotion(bool blocked=true)			const	;
    bool	isCompleted()					const	;
    bool	GoRestPosture()					const	;
    void	ReverseRotation(bool isLeft)			const	;
    void	EnableRelativePositionConstraint(bool on)	const	;
    void	EnableGazeConstraint(bool on)			const	;

  // ForwardKinematicsService
    bool	SelectBaseLink(const char* linkname)		const	;
    bool	GetReferencePose(const char* linkname,
				 TimedPose& D)			const	;
    bool	GetRealPose(const char* linkName, TimedPose& D)	const	;
    bool	GetRealPose(const char* linkName,
			    Time time, TimedPose& D)		const	;

  // SequencePlayerService
    void	go_halfsitting()					;
    void	go_clothinit()						;
    void	go_pickupclothtableinitpos()				;
    void	go_leftpickupclothtableinitpos()			;
    void	go_rightpickupclothtableinitpos()			;
    void	go_leftpushhangclothpos()				;
    void	go_leftlowpushhangclothpos()				;
    void	go_leftpushhangclothpos2()				;
    void	go_lefthangclothpos()					;
    void	go_righthangclothpos()					;
    void	go_handopeningpos(bool isLeft, double ang)	const	;
    void	go_leftarmpos(double angle)			const	;
    void	chest_rotate(int yaw, int pitch)			;
    void	head_rotate(int yaw, int pitch)				;

  // WalkGeneratorService
    void	walkTo(double x, double y, double theta)	const	;
    void	arcTo(double x, double y, double theta)		const	;

  private:
    bool	init(int argc, char* argv[])				;
    template <class SERVICE> typename SERVICE::_ptr_type
		getService(const std::string& name,
			   CORBA::ORB_ptr orb, RTC::CorbaNaming* naming);
    bool	getServiceIOR(RTC::CorbaConsumer<RTC::RTObject> rtc,
			      const std::string& serviceName)		;
    bool	isSuccess(bool success, size_t n, ...)		const	;
    bool	isTrue(bool ret, size_t n, ...)			const	;
    void	seqplay(mask_id id)				const	;
    
  private:
    const char*					_ior;
    OpenHRP::ReachingService_var		_reaching;
    OpenHRP::ReachingService::motion_var	_motion;
    OpenHRP::SequencePlayerService_var		_seqplayer;
    OpenHRP::ForwardKinematicsService_var	_fk;
    OpenHRP::WalkGeneratorService_var		_walkgenerator;

    OpenHRP::dSequence				_posture[4];
    OpenHRP::bSequence				_mask[5];

    GetRealPoseThread				_getRealPose;
    ExecuteCommandThread			_executeCommand;

    bool					_verbose;
};

}
#endif	// !__TUHRP2PP_H
