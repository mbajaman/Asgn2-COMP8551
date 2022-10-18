#include "Position.h"

Position::Position(void)
{
	setPosition(0.0, 0.0);
	m_vx = m_vy = 0.0;
}

Position::~Position(void)
{
}

void Position::applyForce(float fx, float fy)
{
	// modify the velocity, assume that the object's mass is 1
	m_vx += fx;
	m_vy += fy;
}

