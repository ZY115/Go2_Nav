//
// Created by tlab-uav on 24-9-6.
//

#ifndef ENUMCLASS_H
#define ENUMCLASS_H

enum class FSMStateName
{
    // EXIT,
    INVALID,
    PASSIVE,
    FIXEDDOWN,
    FIXEDSTAND,
    FREESTAND,
    TROTTING,

    SWINGTEST,
    BALANCETEST,

    OCS2,
    RL
};

enum class FSMMode
{
    NORMAL,
    CHANGE
};

enum class FrameType
{
    HIP,
    BODY,
    WORLD
};

enum class PhaseStatus
{
    STANCE_ALL,
    SWING_ALL,
    WAVE_ALL
};

#endif //ENUMCLASS_H
