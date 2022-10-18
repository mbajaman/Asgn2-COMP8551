#pragma once

#include <math.h>
#include <stdlib.h>

#define SQR(x) ((x) * (x))

class Position
{
protected:
	float m_x, m_y;
	float m_vx, m_vy;
public:
	Position(void);
	~Position(void);
	void setPosition(float x, float y) { m_x = x; m_y = y; }
	void setRandomPosition(float maxx, float maxy) {
		m_x = rand() * maxx / float(RAND_MAX);
		m_y = rand() * maxy / float(RAND_MAX);
	}

	void setRandomVelocity(float maxx, float maxy)	{
		m_vx = rand() * maxx / float(RAND_MAX);
		m_vy = rand() * maxy / float(RAND_MAX);
	}
	float x() const { return m_x; }
	float y() const { return m_y; }
	float dist(const Position & p) const { return sqrt(SQR(m_x - p.m_x) + SQR(m_y - p.m_y)); } 
	float vx() const { return m_vx; }
	float vy() const { return m_vy; }
	void applyForce(float fx, float fy);
	void move(float dt) { m_x += m_vx * dt; m_y += m_vy * dt; }
	void setVelocity(float vx, float vy) { m_vx = vx; m_vy = vy; }
};
