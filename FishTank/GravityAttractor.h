#ifndef _GRAVITY_ATTRACTOR_H
#define _GRAVITY_ATTRACTOR_H

class GravityAttractor : public Attractor {
private:
	float m_k;
public:
	GravityAttractor(float k) { m_k = k; }
	virtual void f(const Position & p, float *fx, float *fy) {
		*fx = 0.0f;
		// f = k * x;
		*fy = -m_k * (p.y() - m_y);
	}

};
#endif