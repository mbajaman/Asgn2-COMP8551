#ifndef _SPRING_ATTRACTOR_H
#define _SPRINT_ATTRACTOR_H

#include "Attractor.h"

class SpringAttractor : public Attractor {
	float m_k; // the spring constant
public:
	SpringAttractor(float k) { m_k = k; }
	virtual void f(const Position & p, float *fx, float *fy) {
		// f = k * x;
		*fx = -m_k * (p.x() - m_x);
		*fy = -m_k * (p.y() - m_y);
	}

};
#endif