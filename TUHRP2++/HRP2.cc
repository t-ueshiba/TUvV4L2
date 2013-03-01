/*
  for the communication between HRP2 control RTC's (currently ReachingRTC and 
  SequencePlayerRTC) service provider and Vision program.
  by nkita 100808

  revise 100812 to add SeqencePlayerService
  revise 100826 to cope with HRP2DOF7 model
  revise 100916 to cope with hand open/close motion
  revise 100921 to cope with new interface of ReachingRTC
  revise 101217 to cope with new interface of ReachingRTC
  revise 110216 to cope with new interface of SequencePlayerRTC
  revise 120523 to cope with exception yeiled by CsNaming etc.
  revise 120724 to cope with new interface of ReachingRTC
  revise 120803 to cope with new interface of ReachingRTC
		and to be compatible with ReachingService.idl
*/

#include "TU/HRP2++.h"
#include <cstdlib>
#include <cstdarg>
#include <boost/foreach.hpp>

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
template <class ITER, class SEQ> static void
copyToSeq(ITER src, SEQ& seq, size_t n)
{
    seq.length(n);
    for (size_t i = 0; i < n; ++i)
	seq[i] = *src++;
}

template <class SEQ, class ITER> static void
copyFromSeq(SEQ& seq, ITER dst, size_t n)
{
    for (size_t i = 0; i < n; ++i)
	*dst++ = seq[i];
}

static inline HRP2::Time
usec(HRP2::Time sec, HRP2::Time nsec)
{
    return 1000000*sec + (nsec + 500)/1000;
}

/************************************************************************
*  class HRP2								*
************************************************************************/
HRP2::HRP2(int argc, char* argv[],
	   bool isLaterMode, const char* linkName, u_int capacity)
    :_reaching(0), _motion(0), _seqplayer(0), _fk(0), _walkgenerator(0),
     _walk_flag(false), _ior(0), _verbose(false),
     _getRealPose(*this, linkName, capacity),
     _executeCommand(*this)
{
    using namespace	std;
    
    if (!init(argc, argv))
	throw runtime_error("HRP2Client::init() failed!");
    setup(false, isLaterMode);
}

bool
HRP2::init(int argc, char* argv[])
{
    using namespace	OpenHRP;
    using namespace	std;
    using		::optarg;
    using		::optind;
    
    const char*		nameServerHost = "localhost:2809";
    optind = 1;
    _verbose = false;
    for (int c; (c = getopt(argc, argv, "N:V")) != EOF; )
	switch (c)
	{
	  case 'N':
	    nameServerHost = optarg;
	    break;
	  case 'V':
	    _verbose = true;
	    break;
	}

  // ORBを取得
    CORBA::ORB_var	orb = CORBA::ORB_init(argc, argv);;

  // NamingServerを取得
    RTC::CorbaNaming*	naming;
    try
    {
	naming = new RTC::CorbaNaming(orb, nameServerHost);
	if (_verbose)
	    cerr << "TU::HRP2: Succeeded to get naming server "
		 << nameServerHost << endl;
    }
    catch (...)
    {
	cerr << "TU::HRP2: FAILED to get naming server." << endl;
	return false;
    }
    
  // ReachingServiceを取得
    _reaching = getService<ReachingService>("Reaching", orb, naming);
    
  // SequencePlayerServiceを取得
    _seqplayer = getService<SequencePlayerService>("SequencePlayer",
						   orb, naming);

  // ForwardKinematicsServiceを取得
    _fk = getService<ForwardKinematicsService>("ForwardKinematics",
					       orb, naming);

  // WalkGeneratorServiceを取得
    _walkgenerator = getService<WalkGeneratorService>("WalkGenerator",
						      orb, naming);
    _walk_flag = (_walkgenerator != 0);

  // set initial _posture
    _posture[INITPOS].length(DOF);
    _posture[HALFSIT].length(DOF);
    _posture[CLOTHINIT].length(DOF);
    _posture[DESIREDPOS].length(DOF);
	
    for (int i = 0; i < DOF; i++)
	_posture[INITPOS][i] = _posture[HALFSIT][i] = 0;

    _posture[INITPOS][R_HAND_P] = deg2rad( 5);
    _posture[INITPOS][L_HAND_P] = deg2rad(-5);

  // set half sitting posture
    _posture[HALFSIT][RLEG_JOINT2] = HALF_SITTING_HIP_ANGLE;
    _posture[HALFSIT][RLEG_JOINT3] = HALF_SITTING_KNEE_ANGLE;
    _posture[HALFSIT][RLEG_JOINT4] = HALF_SITTING_ANKLE_ANGLE;

    _posture[HALFSIT][LLEG_JOINT2] = HALF_SITTING_HIP_ANGLE;
    _posture[HALFSIT][LLEG_JOINT3] = HALF_SITTING_KNEE_ANGLE;
    _posture[HALFSIT][LLEG_JOINT4] = HALF_SITTING_ANKLE_ANGLE;

    _posture[HALFSIT][RARM_JOINT0] = deg2rad( 15);
    _posture[HALFSIT][RARM_JOINT1] = deg2rad(-10);
    _posture[HALFSIT][RARM_JOINT3] = deg2rad(-30);
    _posture[HALFSIT][R_HAND_P] = deg2rad( 10);

    _posture[HALFSIT][LARM_JOINT0] = deg2rad( 15);
    _posture[HALFSIT][LARM_JOINT1] = deg2rad( 10);
    _posture[HALFSIT][LARM_JOINT3] = deg2rad(-30);
    _posture[HALFSIT][L_HAND_P] = deg2rad(-10);

    for (int i = 0;i < DOF; i++)
	_posture[CLOTHINIT][i] = _posture[DESIREDPOS][i] = _posture[HALFSIT][i];
	
    _posture[CLOTHINIT][RARM_JOINT0] = deg2rad(30);
    _posture[CLOTHINIT][RARM_JOINT1] = 0.0;
  //  _posture[CLOTHINIT][RARM_JOINT3] = deg2rad(-105);
    _posture[CLOTHINIT][RARM_JOINT3] = deg2rad(-100);
    _posture[CLOTHINIT][RARM_JOINT4] = 0.0;
    _posture[CLOTHINIT][RARM_JOINT5] = deg2rad(-15);

  // set initial mask
    _mask[HANDS].length(DOF);
    _mask[LEFTHAND].length(DOF);
    _mask[RIGHTHAND].length(DOF);
    _mask[DESIREDMASK].length(DOF);
    _mask[EXCEPTHEAD].length(DOF);
	
    for (int i = 0; i < DOF; i++)
    {
	_mask[HANDS][i] = _mask[LEFTHAND][i]
		       = _mask[RIGHTHAND][i] = _mask[DESIREDMASK][i] = 1;
	_mask[EXCEPTHEAD][i] = 0;
    }

    _mask[HANDS][L_HAND_P] = _mask[LEFTHAND][L_HAND_P] = 0;
    _mask[HANDS][R_HAND_P] = _mask[RIGHTHAND][R_HAND_P] = 0;
    _mask[EXCEPTHEAD][HEAD_JOINT0] = _mask[EXCEPTHEAD][HEAD_JOINT1] = 1;

    return isSuccess(true, 1, " to initialize HRP2.");
}

/*
 *  functions for ReachingService
 */
bool
HRP2::SelectArm(bool isLeft) const
{
    using namespace	OpenHRP;

    return isSuccess(_reaching->selectArm(isLeft ?
					  ReachingService::LEFT_ARM :
					  ReachingService::RIGHT_ARM),
		     3,
		     " to select ", (isLeft ? "left" : "right"), " arm.");
}

bool
HRP2::DeSelectArm(bool isLeft) const
{
    using namespace	OpenHRP;

    return isSuccess(_reaching->deselectArm(isLeft ?
					    ReachingService::LEFT_ARM :
					    ReachingService::RIGHT_ARM),
		     3,
		     " to deselect ", (isLeft ? "left" : "right"), " arm.");
}

bool
HRP2::isSelected(bool isLeft) const
{
    using namespace	OpenHRP;

    return isTrue(_reaching->isSelected(isLeft ? ReachingService::LEFT_ARM
					       : ReachingService::RIGHT_ARM),
		  3, " selected", (isLeft ? "left" : "right"), " arm.");
}

bool
HRP2::isMaster(bool isLeft) const
{
    using namespace	OpenHRP;

    return isTrue(_reaching->isMaster(isLeft ? ReachingService::LEFT_ARM
					     : ReachingService::RIGHT_ARM),
		  3, " a master", (isLeft ? "left" : "right"), " arm");
}

bool
HRP2::SetGraspCenter(bool isLeft, const double* pos) const
{
    using namespace	OpenHRP;

    ReachingService::dsequence dpos;
    copyToSeq(pos, dpos, 3);

    return isSuccess(_reaching->setGraspCenter((isLeft ?
						ReachingService::LEFT_ARM :
						ReachingService::RIGHT_ARM),
					       dpos),
		     3, " to set GraspCenter for ",
		     (isLeft ? "left" : "right"), " arm.");
}

bool
HRP2::SelectUsedDofs(const bool* used) const
{
    using namespace	OpenHRP;

    ReachingService::bsequence	bused;
    copyToSeq(used, bused, DOF + 6);

    return isSuccess(_reaching->selectUsedDofs(bused),
		     1, " to select used DOFs.");
}

bool
HRP2::SelectTaskDofs(bool isLeft,
		     const bool* constrained, const double* weights) const
{
    using namespace	OpenHRP;

    ReachingService::bsequence	bools;
    copyToSeq(constrained, bools, 6);

    ReachingService::dsequence	wei;
    copyToSeq(weights, wei, 6);

    return isSuccess(_reaching->selectTaskDofs((isLeft ?
						ReachingService::LEFT_ARM :
						ReachingService::RIGHT_ARM),
					       bools, wei),
		     3, " to select task DOF for ",
		     (isLeft ? "left" : "right"), " arm.");
}

bool
HRP2::SelectExecutionMode(bool isLaterMode) const
{
    using namespace	OpenHRP;

    return isSuccess(_reaching->selectExecutionMode(
				  isLaterMode ? ReachingService::LATER
					      : ReachingService::IMMEDIATELY),
		     3, " to select ",
		     (isLaterMode ? "LATER" : "IMMEDIATELY"), " mode.");
} 

bool
HRP2::SetTargetPose(bool isLeft, const double* pose, double duration) const
{
    using namespace	OpenHRP;

    ReachingService::dsequence	hmat;
    copyToSeq(pose, hmat, 16);

    return isSuccess(_reaching->setTargetPose((isLeft ?
					       ReachingService::LEFT_ARM :
					       ReachingService::RIGHT_ARM),
					      hmat, duration),
		     3, " to set target pose for ",
		     (isLeft ? "left" : "right"), " arm.");
}

bool
HRP2::SetTargetArc(bool isLeft,
		   const double* rot, const double* cen,
		   const double* ax, double theta, double duration) const
{
    using namespace	OpenHRP;

    ReachingService::dsequence rotation;
    copyToSeq(rot, rotation, 9);
    ReachingService::dsequence center;
    copyToSeq(cen, center, 3);
    ReachingService::dsequence axis;
    copyToSeq(ax, axis, 3);

    return isSuccess(_reaching->setTargetArc((isLeft ?
					      ReachingService::LEFT_ARM :
					      ReachingService::RIGHT_ARM),
					     rotation, center, axis, theta,
					     duration),
		     3, " to set target arc for ", 
		     (isLeft ? "left" : "right"), " arm.");
}

bool
HRP2::SetTargetVelocity(bool isLeft,
			const double* velocity, double duration) const
{
    using namespace	OpenHRP;

    ReachingService::dsequence	vel;
    copyToSeq(velocity, vel, 6);

    return isSuccess(_reaching->setTargetVelocity(
				    (isLeft ? ReachingService::LEFT_ARM
					    : ReachingService::RIGHT_ARM),
				    vel, duration),
		     3, " to set target velocity for ",
		     (isLeft ? "left" : "right"), " arm.");
}

bool
HRP2::GetCurrentPose(bool isLeft, double* pose) const
{
    using namespace	OpenHRP;

    ReachingService::dsequence_var	hmat;
    bool ret = _reaching->getCurrentPose((isLeft ?
					  ReachingService::LEFT_ARM :
					  ReachingService::RIGHT_ARM),
					 hmat);
    if (ret)
	copyFromSeq(hmat, pose, 16);

    return isSuccess(ret, 3, " to get current pose for ",
		     (isLeft ? "left" : "right"), " arm.");
}

void
HRP2::GenerateMotion(bool blocked) const
{
    if (blocked)
    {
	_reaching->generateMotion();
	isSuccess(true, 1, " to do generateMotion.");
    }
    else
	_executeCommand(&HRP2::GenerateMotion);
} 

void
HRP2::PlayMotion(bool blocked) const
{
    if (blocked)
    {
	isSuccess(_reaching->playMotion(), 1, " to do playMotion.");
    }
    else
	_executeCommand(&HRP2::PlayMotion);
} 

bool
HRP2::isCompleted() const
{
    return _executeCommand.isCompleted();
}
    
bool
HRP2::GoRestPosture() const
{
    return isSuccess(_reaching->goRestPosture(), 1, " to do goRestPosture.");
} 

void
HRP2::getMotion()
{
    _motion = _reaching->getMotion();
}

u_int
HRP2::getMotionLength() const
{
    return _motion->length();
}

bool
HRP2::getPosture(u_int rank, double*& q, double*& p, double*& rpy, double*& zmp)
{
    if (rank > _motion->length())
	return false;

    q   = _motion[rank].q.get_buffer();
    p   = _motion[rank].p.get_buffer();
    rpy = _motion[rank].rpy.get_buffer();
    zmp = _motion[rank].zmp.get_buffer();
    return true;
}

void
HRP2::ReverseRotation(bool isLeft) const
{
    using namespace	OpenHRP;

    _reaching->reverseRotation(isLeft ? ReachingService::LEFT_ARM
				      : ReachingService::RIGHT_ARM);
    isSuccess(true, 3, " to do ReverseRotation for ",
	      (isLeft ? "left" : "right"), " arm.");
} 

void
HRP2::EnableRelativePositionConstraint(bool on) const
{
    _reaching->enableRelativePositionConstraint(on);
    isTrue(on, 1, " enabled relative position constraint.");
} 

void
HRP2::EnableGazeConstraint(bool on) const
{
    _reaching->enableGazeConstraint(on);
    isTrue(on, 1, " enabled gaze constraint.");
} 

/*
 *  functions for ForwardKinematicsService
 */
bool
HRP2::SelectBaseLink(const char* linkname) const
{
    return isSuccess(_fk->selectBaseLink(linkname),
		     1, " to select BaseLink.");
} 

bool
HRP2::GetReferencePose(const char* linkname, TimedPose& D) const
{
    RTC::TimedDoubleSeq_var	hmat;
    bool			success = _fk->getReferencePose(linkname,
								hmat);
    if (success)
    {
	copyFromSeq(hmat->data, D.data(), D.nrow() * D.ncol());
	D.t = usec(hmat->tm.sec, hmat->tm.nsec);
    }

    return isSuccess(success, 1, " to get reffrence pose.");
}

bool
HRP2::GetRealPose(const char* linkname, Time time, TimedPose& D) const
{
    return _getRealPose(time, D);
}

bool
HRP2::GetRealPose(const char* linkname, TimedPose& D) const
{
    RTC::TimedDoubleSeq_var	hmat;
    bool			success = _fk->getCurrentPose(linkname, hmat);
    if (success)
    {
	copyFromSeq(hmat->data, D.data(), D.nrow() * D.ncol());
	D.t = usec(hmat->tm.sec, hmat->tm.nsec);
    }

  //return isSuccess(success, 1, " to get real pose.");
    return success;
}

/*
 *  functions for SequencePlayerService
 */
void
HRP2::go_halfsitting()
{
    for (int i = 0; i < DOF; i++)
	_posture[DESIREDPOS][i] = _posture[HALFSIT][i];

    seqplay(HANDS);
}

void
HRP2::go_clothinit()
{
    for (int i = 0; i < DOF; i++)
	_posture[DESIREDPOS][i] = _posture[CLOTHINIT][i];

    seqplay(HANDS);
}

void
HRP2::go_pickupclothtableinitpos()
{
    go_leftpickupclothtableinitpos();
}

void
HRP2::go_leftpickupclothtableinitpos()
{
    _posture[DESIREDPOS][LARM_JOINT0] = deg2rad(-105);
    _posture[DESIREDPOS][LARM_JOINT1] = deg2rad(42);
    _posture[DESIREDPOS][LARM_JOINT2] = deg2rad(-72);
    _posture[DESIREDPOS][LARM_JOINT3] = deg2rad(-106);
    _posture[DESIREDPOS][LARM_JOINT4] = deg2rad(25);
    _posture[DESIREDPOS][LARM_JOINT5] = deg2rad(0);
    _posture[DESIREDPOS][LARM_JOINT6] = deg2rad(45); // added by nkita 110214

    seqplay(HANDS);
}

void
HRP2::go_rightpickupclothtableinitpos()
{
    _posture[DESIREDPOS][RARM_JOINT0] = deg2rad(-105);
    _posture[DESIREDPOS][RARM_JOINT1] = deg2rad(-42);
    _posture[DESIREDPOS][RARM_JOINT2] = deg2rad(72);
    _posture[DESIREDPOS][RARM_JOINT3] = deg2rad(-106);
    _posture[DESIREDPOS][RARM_JOINT4] = deg2rad(-25);
    _posture[DESIREDPOS][RARM_JOINT5] = deg2rad(0);
    _posture[DESIREDPOS][RARM_JOINT6] = deg2rad(45);

    seqplay(HANDS);
}

void
HRP2::go_leftpushhangclothpos()
{
    _posture[DESIREDPOS][LARM_JOINT0] = deg2rad(-72.6);
    _posture[DESIREDPOS][LARM_JOINT1] = deg2rad(95);
    _posture[DESIREDPOS][LARM_JOINT2] = deg2rad(-81);
    _posture[DESIREDPOS][LARM_JOINT3] = deg2rad(-136.8);
    _posture[DESIREDPOS][LARM_JOINT4] = deg2rad(82.8);
    _posture[DESIREDPOS][LARM_JOINT5] = deg2rad(0.0);
    _posture[DESIREDPOS][L_HAND_P]    = deg2rad(0.0);

    seqplay(HANDS);
}

void
HRP2::go_leftlowpushhangclothpos()
{
    _posture[DESIREDPOS][LARM_JOINT0] = deg2rad(-35.8);
    _posture[DESIREDPOS][LARM_JOINT1] = deg2rad(61.4);
    _posture[DESIREDPOS][LARM_JOINT2] = deg2rad(-80.9);
    _posture[DESIREDPOS][LARM_JOINT3] = deg2rad(-137);
    _posture[DESIREDPOS][LARM_JOINT4] = deg2rad(46.4);
    _posture[DESIREDPOS][LARM_JOINT5] = deg2rad(-7.4);
    _posture[DESIREDPOS][L_HAND_P]    = deg2rad(0);

    seqplay(HANDS);
}

void
HRP2::go_leftpushhangclothpos2()
{
    _posture[DESIREDPOS][LARM_JOINT0] = deg2rad(-80);
    _posture[DESIREDPOS][LARM_JOINT1] = deg2rad(60);
    _posture[DESIREDPOS][LARM_JOINT2] = deg2rad(-90);
    _posture[DESIREDPOS][LARM_JOINT3] = deg2rad(-130);
    _posture[DESIREDPOS][LARM_JOINT4] = deg2rad(0);
    _posture[DESIREDPOS][LARM_JOINT5] = deg2rad(0.0);
    _posture[DESIREDPOS][L_HAND_P]    = deg2rad(0.0);

    seqplay(HANDS);
}

void
HRP2::go_lefthangclothpos()
{
    _posture[DESIREDPOS][LARM_JOINT0] = deg2rad(-145);
    _posture[DESIREDPOS][LARM_JOINT1] = deg2rad(42.5);
    _posture[DESIREDPOS][LARM_JOINT2] = deg2rad(-90);
    _posture[DESIREDPOS][LARM_JOINT3] = deg2rad(-99);
    _posture[DESIREDPOS][LARM_JOINT4] = deg2rad(59);
    _posture[DESIREDPOS][LARM_JOINT5] = deg2rad(58.8);

    seqplay(HANDS);
}

void
HRP2::go_righthangclothpos()
{
    _posture[DESIREDPOS][RARM_JOINT0] = deg2rad(-145);
    _posture[DESIREDPOS][RARM_JOINT1] = deg2rad(-42.5);
    _posture[DESIREDPOS][RARM_JOINT2] = deg2rad(90);
    _posture[DESIREDPOS][RARM_JOINT3] = deg2rad(-99);
    _posture[DESIREDPOS][RARM_JOINT4] = deg2rad(-59);
    _posture[DESIREDPOS][RARM_JOINT5] = deg2rad(58.8);

    seqplay(HANDS);
}

void
HRP2::go_handopeningpos(bool isLeft, double angle) const
{
    using namespace	std;
    
    if (_verbose)
	cerr << "TU::HRP2: Begin to send HandOpening posture angles to SequencePlayerRTC." << endl;

    double time = 5.0;
    _seqplayer->setJointAngle((isLeft ? LARM_HAND : RARM_HAND),
			      deg2rad(isLeft ? angle : -angle), time);
    while (!_seqplayer->isEmpty())
	;

    if (_verbose)
	cerr << "TU::HRP2: Finish to send HandOpening posture angles to SequencePlayerRTC." << endl;
}

void
HRP2::go_leftarmpos(double angle) const
{
    using namespace	std;

    if (_verbose)
	cerr << "TU::HRP2: Begin to send LARM_JOINT6 angle to SequencePlayerRTC." << endl;

    double time = 5.0;
    _seqplayer->setJointAngle("LARM_JOINT6", deg2rad(angle), time);
    while (!_seqplayer->isEmpty())
	;
    
    if (_verbose)
	cerr << "TU::HRP2: Finish to send LARM_JOINT6 angle to SequencePlayerRTC." << endl;
}

void
HRP2::chest_rotate(int yaw, int pitch)
{
    _posture[DESIREDPOS][CHEST_JOINT0] = deg2rad(yaw);
    _posture[DESIREDPOS][CHEST_JOINT1] = deg2rad(pitch);

    seqplay(EXCEPTHEAD);
}

void
HRP2::head_rotate(int yaw, int pitch)
{
    _posture[DESIREDPOS][HEAD_JOINT0] = deg2rad(yaw);
    _posture[DESIREDPOS][HEAD_JOINT1] = deg2rad(pitch);

    seqplay(EXCEPTHEAD);
}

/*
 *  functions for WalkGeneratorService
 */
void
HRP2::walkTo(double x, double y, double theta) const
{
    if (_walk_flag)
    {
	_walkgenerator->setTargetPosNoWait(x, y, deg2rad(theta));
	isSuccess(true, 1, " to do walkTo.");
    }
    return;
} 

void
HRP2::arcTo(double x, double y, double theta) const
{
    if (_walk_flag)
    {
	_walkgenerator->setArcNoWait(x, y, deg2rad(theta));
	isSuccess(true, 1, " to do arcTo.");
    }
    return;
} 

/*
 *  private member functions
 */
template <class SERVICE> typename SERVICE::_ptr_type
HRP2::getService(const std::string& name,
		 CORBA::ORB_ptr orb, RTC::CorbaNaming* naming)
{
    using namespace	OpenHRP;
    using namespace	std;
    
  // RTCを取得
    RTC::CorbaConsumer<RTC::RTObject>	rtc;
    try
    {
	rtc.setObject(naming->resolve((name + "0.rtc").c_str()));
	rtc->_non_existent();
	if (_verbose)
	    cerr << "TU::HRP2: Succeeded to get RTC of " << name
		 << '.' << endl;
    }
    catch (...)
    {
	cerr << "TU::HRP2: FAILED to get RTC of " << name << '.' << endl;
	return 0;
    }

  // Serviceを取得
    if (!getServiceIOR(rtc, std::string((name + "Service").c_str())))
    {
	cerr << "TU::HRP2: FAILED to get IOR of " << name << "Service."
	     << endl;
	return 0;
    }
    if (_verbose)
	cerr << "TU::HRP2: Succeeded to get " << name << "Service." << endl;

    return SERVICE::_narrow(orb->string_to_object(_ior));
}

bool
HRP2::getServiceIOR(RTC::CorbaConsumer<RTC::RTObject> rtc,
		    const std::string& serviceName)
{
  // TargetRTCのポートリストを取得
    RTC::PortServiceList	ports = *(rtc->get_ports());
    if (ports.length() <= 0)
	return isSuccess(false, 1, " to get PortServiceList of RTC.");

    RTC::ComponentProfile*	cprof = rtc->get_component_profile();
    std::string			portname = std::string(cprof->instance_name)
					 + "." + serviceName;

    for (u_int i = 0; i < ports.length(); i++)
    {
	RTC::PortService_var	port = ports[i];
	RTC::PortProfile*	prof = port->get_port_profile();
	if (std::string(prof->name) == portname)
	{
	    RTC::ConnectorProfile	connProfile;
	    connProfile.name	     = "noname";
	    connProfile.connector_id = "";
	    connProfile.ports.length(1);
	    connProfile.ports[0]     = port;
	    connProfile.properties   = NULL;
	    port->connect(connProfile);

	    connProfile.properties[0].value >>= _ior;

	    port->disconnect(connProfile.connector_id);
	    
	    return true;
	}
    }

    return false;
}

bool
HRP2::isSuccess(bool success, size_t n, ...) const
{
    using namespace	std;
    
    if (_verbose || !success)
    {
	va_list	args;
	va_start(args, n);
    
	cerr << "TU::HRP2:" << (success ? " succeeded" : " FAILED");
	for (size_t i = 0; i < n; ++i)
	{
	    const char*	s = va_arg(args, const char*);
	    cerr << s;
	}
	cerr << endl;

	va_end(args);
    }
    
    return success;
}

bool
HRP2::isTrue(bool success, size_t n, ...) const
{
    using namespace	std;
    
    if (_verbose || !success)
    {
	va_list	args;
	va_start(args, n);
    
	cerr << "TU::HRP2:" << (success ? " " : " NOT");
	for (size_t i = 0; i < n; ++i)
	{
	    const char*	s = va_arg(args, const char*);
	    cerr << s;
	}
	cerr << endl;

	va_end(args);
    }
    
    return success;
}

void
HRP2::setup(bool isLeftHand, bool isLaterMode)
{
    using namespace	std;
    
  // 拘束設定
    bool	constrained[] = {true, true, true, true, true, true};
    double	weights[]     = {10.0, 10.0, 10.0, 10.0, 10.0, 10.0};
    if (!SelectTaskDofs(isLeftHand, constrained, weights))
	throw runtime_error("HRP2Client::SelectTaskDofs() failed!!");
    
  // 使用する自由度を設定
    bool	usedDofs[] =
		{
		    true, true, true, true, true, true,		// right leg
		    true, true, true, true, true, true,		// left leg
		    true, false,				// waist
		    false, false,				// neck
		    true, true, true, true, true, true, true,	// right arm
		    false,					// right hand
		    false, false, false, false, false, false, false, // left arm
		    false,					// left hand
		    true, true, true, false, false, false	// base
		};
    if (!SelectUsedDofs(usedDofs))
	throw runtime_error("HRP2Client::SelectUsedDofs() failed!!");
    
  // ベースとなるリンクを設定
    if (!SelectBaseLink("RLEG_JOINT5"))
	throw runtime_error("HRP2Client::SelectBaseLink() failed!!");

  // LaterモードまたはImmediateモードに設定
    if (!SelectExecutionMode(isLaterMode))
	throw runtime_error("HRP2Client::SelectExecutionMode() failed!!");

  // 反対側の手を強制的にdeselectし，所望の手をselectする．
    DeSelectArm(!isLeftHand);
    SelectArm(isLeftHand);

  // スレッドを起動する．
    _getRealPose.run();
    _executeCommand.run();
}

void
HRP2::seqplay(mask_id id) const
{
    using namespace	OpenHRP;
    using namespace	std;

    if (_verbose)
	cerr << "TU::HRP2: Begin to send posture angles to SequencePlayerRTC."
	     << endl;
    
    double	time = 5.0;
    if (!_seqplayer->setJointAnglesWithMask(_posture[DESIREDPOS],
					    _mask[id], time))
	cerr << "TU::HRP2: setJointAnglesWithMask() FAILED." << endl;

    dSequence	waistpos;
    waistpos.length(3);
    waistpos[0] = waistpos[1] = 0.0;
    waistpos[2] = HALF_SITTING_WAIST_HEIGHT;
    if (!_seqplayer->setBasePos(waistpos, time))
	cerr << "TU::HRP2: setBasePos() FAILED." << endl;

    dSequence	waistrpy;
    waistrpy.length(3);
    waistrpy[0] = waistrpy[1] = waistrpy[2] = 0.0;
    if (!_seqplayer->setBaseRpy(waistrpy, time))
	cerr << "TU::HRP2: setBaseRpy() FAILED." << endl;

    dSequence	zmp;
    zmp.length(3);
    zmp[0] = zmp[1] = 0.0;
    zmp[2] = -HALF_SITTING_WAIST_HEIGHT;
    if (!_seqplayer->setZmp(zmp, time))
	cerr << "TU::HRP2: setZmp() FAILED." << endl;

    _seqplayer->waitInterpolation();
    
    if (_verbose)
	cerr << "TU::HRP2: Finish to send posture angles to SequencePlayerRTC."
	     << endl;
}

/************************************************************************
*  class HRP2::GetRealPoseThread					*
************************************************************************/
HRP2::GetRealPoseThread::GetRealPoseThread(HRP2& hrp2,
					   const char* linkName,
					   u_int capacity)
    :_hrp2(hrp2), _linkName(linkName), _poses(capacity),
     _quit(false), _mutex(), _thread()
{
    pthread_mutex_init(&_mutex, NULL);
}

HRP2::GetRealPoseThread::~GetRealPoseThread()
{
    pthread_mutex_lock(&_mutex);
    _quit = true;			// 終了フラグを立てる
    pthread_mutex_unlock(&_mutex);

    pthread_join(_thread, NULL);	// 子スレッドの終了を待つ
    pthread_mutex_destroy(&_mutex);
}

void
HRP2::GetRealPoseThread::run()
{
    pthread_create(&_thread, NULL, threadProc, this);

    usleep(1000);
}
    
bool
HRP2::GetRealPoseThread::operator ()(Time time, TimedPose& D) const
{
  // 与えられた時刻よりも後のポーズが得られるまで待つ．
    TimedPose	after;		// timeよりも後の時刻で取得されたポーズ
    for (;;)			// を発見するまで待つ．
    {
	pthread_mutex_lock(&_mutex);
	if (!_poses.empty() && (after = _poses.back()).t > time)
	    break;
	pthread_mutex_unlock(&_mutex);
    }

  // リングバッファを過去に遡り，与えられた時刻よりも前のポーズを探す．
    BOOST_REVERSE_FOREACH (const TimedPose& pose, _poses)
    {
	if (pose.t <= time)	// timeの直前のポーズならば．．．
	{
	  // poseとafterのうち，その時刻がtimeに近い方を返す．
	    if ((time - pose.t) < (after.t - time))
	    {			// timeがafterよりもposeの時刻に近ければ．．．
		D = pose;
	    }
	    else		// timeがposeよりもafterの時刻に近ければ．．．
	    {
		D = after;
	    }
	    pthread_mutex_unlock(&_mutex);

	    return true;	// timeを挟む２つのポーズを発見した
	}
	after = pose;
    }
    pthread_mutex_unlock(&_mutex);

    return false;		// timeの直前のポーズを発見できなかった
}

void
HRP2::GetRealPoseThread::timeSpan(Time& t0, Time& t1) const
{
    for (;;)
    {
	pthread_mutex_lock(&_mutex);
	if (!_poses.empty())
	{
	    t0 = _poses.front().t;
	    t1 = _poses.back().t;
	    pthread_mutex_unlock(&_mutex);
	    break;
	}
	pthread_mutex_unlock(&_mutex);
    }
}
    
void*
HRP2::GetRealPoseThread::mainLoop()
{
    for (;;)
    {
	pthread_mutex_lock(&_mutex);
	bool	quit = _quit;		// 命令を取得
	pthread_mutex_unlock(&_mutex);
	if (quit)			// 終了命令ならば...
	    break;			// 脱出

	TimedPose	D;
	if (_hrp2.GetRealPose(const_cast<char*>(_linkName.c_str()), D))
	{					// ポーズ入力成功？
	    if (_poses.empty() || (D.t != _poses.back().t))
	    {
		pthread_mutex_lock(&_mutex);
		_poses.push_back(D);		// リングバッファに入れる
		pthread_mutex_unlock(&_mutex);
	    }
	}
    }

    return 0;
}
    
void*
HRP2::GetRealPoseThread::threadProc(void* thread)
{
    GetRealPoseThread*	th = static_cast<GetRealPoseThread*>(thread);

    return th->mainLoop();
}

/************************************************************************
*  class HRP2::ExecuteCommandThread					*
************************************************************************/
HRP2::ExecuteCommandThread::ExecuteCommandThread(HRP2& hrp2)
    :_hrp2(hrp2), _commands(), _quit(false), _mutex(), _cond(), _thread()
{
    pthread_mutex_init(&_mutex, NULL);
    pthread_cond_init(&_cond, NULL);
}

HRP2::ExecuteCommandThread::~ExecuteCommandThread()
{
    pthread_mutex_lock(&_mutex);
    _quit = true;			// 終了命令をセット
    pthread_cond_signal(&_cond);	// 子スレッドに終了命令を送る
    pthread_mutex_unlock(&_mutex);

    pthread_join(_thread, NULL);	// 子スレッドの終了を待つ
    pthread_cond_destroy(&_cond);
    pthread_mutex_destroy(&_mutex);
}

void
HRP2::ExecuteCommandThread::run()
{
    pthread_create(&_thread, NULL, threadProc, this);

    usleep(1000);
}
    
void
HRP2::ExecuteCommandThread::operator ()(Command command) const
{
    pthread_mutex_lock(&_mutex);
    _commands.push(command);		// コマンドをキューに入れる
    pthread_cond_signal(&_cond);	// 子スレッドに実行命令を送る
    pthread_mutex_unlock(&_mutex);
}

void
HRP2::ExecuteCommandThread::wait() const
{
    pthread_mutex_lock(&_mutex);
    while (!_commands.empty())		// 全コマンドの実行が完了するまで
	pthread_cond_wait(&_cond, &_mutex);	// 待つ
    pthread_mutex_unlock(&_mutex);
}

bool
HRP2::ExecuteCommandThread::isCompleted() const
{
    pthread_mutex_lock(&_mutex);
    bool	empty = _commands.empty();	// キューが空？
    pthread_mutex_unlock(&_mutex);

    return empty;			// キューが空ならば実行完了
}
    
void*
HRP2::ExecuteCommandThread::mainLoop()
{
    pthread_mutex_lock(&_mutex);
    for (;;)
    {
	pthread_cond_wait(&_cond, &_mutex);	// 親からの命令を待つ
	if (_quit)				// スレッド終了命令ならば...
	    break;				// ループを脱出
	else if (!_commands.empty())		// 実行命令ならば...
	{
	    Command	command = _commands.front();	// 最古のコマンド
	    pthread_mutex_unlock(&_mutex);
	    (_hrp2.*command)(true);		// 最古のコマンドを実行
	    pthread_mutex_lock(&_mutex);
	    _commands.pop();			// 最古のコマンドを捨てる
	    pthread_cond_signal(&_cond);	// 親に実行が完了したことを通知
	}
    }
    pthread_mutex_unlock(&_mutex);

    return 0;
}

void*
HRP2::ExecuteCommandThread::threadProc(void* thread)
{
    ExecuteCommandThread* th = static_cast<ExecuteCommandThread*>(thread);

    return th->mainLoop();
}

}
