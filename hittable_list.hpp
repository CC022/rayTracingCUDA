#pragma once
#include <stdio.h>
#include "hittable.hpp"

class hittable_list: public hittable {
public:
    hittable **list;
    int list_size;
    
    __host__ __device__ hittable_list() {}
    __host__ __device__ hittable_list(hittable **l, int n) {list = l; list_size = n;}
    
    __host__ __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        hit_record temp_rec;
        bool hitAnything = false;
        double closestSoFar = t_max;
        for (int i=0; i<list_size; i++) {
            if (list[i]->hit(r, t_min, closestSoFar, temp_rec)) {
                hitAnything = true;
                closestSoFar = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hitAnything;
    }
    
};