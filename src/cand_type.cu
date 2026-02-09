#include "cand_type.hpp"


__host__ __device__
Cand::Cand(double y, double x, int dir, double score) {
    this->y = y;
    this->x = x;
    this->dir = dir;
    this->score = score;
}

__host__ __device__
bool Cand::operator<(const Cand& otherCand) {
    return (score > otherCand.score);
}

__host__ __device__
double Cand::dist(const Cand& cand1, const Cand& cand2) {
    Vec v(cand2.y - cand1.y, cand2.x - cand1.x);
    return v.len();
}

__host__ __device__
int Cand::dirDiff(const Cand& cand1, const Cand& cand2) {
    return (cand1.dir - cand2.dir + DIRECTIONS) % DIRECTIONS;
}

__host__ __device__
double Cand::distToLine(const Cand& otherCand) const {
    Vec v(otherCand.y - y, otherCand.x - x);
    Vec u = getOrthogonalUnitVector(dir);
    Vec p = v.subtract(u*v.dot(u));
    return p.len();
}