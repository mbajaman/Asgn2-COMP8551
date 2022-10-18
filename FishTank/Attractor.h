#ifndef _ATTRACTOR_H_
#define _ATTRACTOR_H_

#include "Position.h"

class Attractor : public Position
{
public:
	// return the amount of force applied on object @ p
	virtual void f(const Position &p, float *fx, float *fy) = 0;
};
#endif