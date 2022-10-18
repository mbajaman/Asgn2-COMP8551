#ifndef WALL_ATTRACTOR_H_
#define WALL_ATTRACTOR_H_

class WallAttractor : public Attractor {
private:
	int m_width, m_height;
public:
	WallAttractor(int width, int height) { m_width = width; m_height = height; }
	virtual void f(const Position & p, float *fx, float *fy) {
		*fx = 0.0f;
		*fy = 0.0f;

		if (p.x() < 0 && p.vx() < 0.0f) *fx = 10.0f;
		if (p.y() < 0 && p.vy() < 0.0f) *fy = 10.0f;
		if (p.x() > m_width && p.vx() > 0.0f) *fx = -10.0f;
		if (p.y() > m_height && p.vy() > 0.0f) *fy = -10.0f; 
	}
};

#endif
