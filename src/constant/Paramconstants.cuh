#pragma once
struct ParamConstants {
    double L1, L2, L3;
    double Unidx, Unidy, Unidz;
    double diam;
    double boltz;
    double Vtl;
    double VHS;
    double Vmax;
    double Vstd;
    double tau;
    size_t Fn;
};
// __constant__ 设备常量内存
extern __device__ __constant__ ParamConstants d_params;