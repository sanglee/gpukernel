#include <curand_kernel.h>

//#include <stdio.h>

//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
//#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
//#endif

__constant__ __device__ long long c_conf[1024]; // 8 KiB ConstMem

#define TWOPI 6.283185307179586

__global__ void initRNG(curandState * const rngStates, const unsigned int n, const unsigned int seed){

    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 

    while(tid < n){
		curand_init(seed, tid, 0, &rngStates[tid]);
		tid += (gridDim.x * blockDim.x);
    }
}

template <class T>
struct GPUConfig{

	unsigned int d;

	T std;
	T scale;

	curandState * rngStates;
	curandState * rngStatesUni;
};

template <class T>
void uploadGPUConfig(const unsigned int d, const T std, const T scale, curandState * const rngStates, curandState * const rngStatesUni ){
	
	GPUConfig<T> *gc = (GPUConfig<T>*) malloc(sizeof(GPUConfig<T>));

	//printf("std = %f, scale = %f\n", std, scale);

	gc->d = d;
	gc->std = std;
	gc->scale = scale;
	gc->rngStates = rngStates;
	gc->rngStatesUni = rngStatesUni;

	cudaMemcpyToSymbol(c_conf, gc, sizeof(GPUConfig<T>));

	free(gc);
}

template void uploadGPUConfig<double>(const unsigned int d, const double std, const double scale, curandState * const rngStates, curandState * const rngStatesUni );
template void uploadGPUConfig<float>(const unsigned int d, const float std, const float scale, curandState * const rngStates, curandState * const rngStatesUni );

/*
template <class T, unsigned int B>
__global__ void doRandomStuff(T* x, T* y, unsigned int n, curandState * rngStates, curandState * rngStatesUni){

	volatile __shared__ T sdata[32];
	unsigned int i = blockIdx.x;

	unsigned int idx = threadIdx.x;
	
	sdata[threadIdx.x] = x[idx] * sqrt(0.002)*curand_normal_double(&rngStates[i*blockDim.x + threadIdx.x]);
		
		if(threadIdx.x < 16){
			sdata[threadIdx.x] += sdata[threadIdx.x + 16];
			sdata[threadIdx.x] += sdata[threadIdx.x +  8];
			sdata[threadIdx.x] += sdata[threadIdx.x +  4];
			sdata[threadIdx.x] += sdata[threadIdx.x +  2];
			sdata[threadIdx.x] += sdata[threadIdx.x +  1];
		}

		if(threadIdx.x == 0) {
			y[i] = sqrt(2./1.) * cos( sdata[0] + 6.283185307179586*curand_uniform_double(&rngStatesUni[i]));
		}

}
*/

template <class T, unsigned int B>
__global__ void doRandomStuff(T* x, T* y, unsigned int n, curandState * rngStates, curandState * rngStatesUni){

	volatile __shared__ T sdata[32];
//	unsigned int i = blockIdx.x;

	unsigned int idx = threadIdx.x;

	curandState mystate, mystate2;
	curand_init(1023, blockIdx.x, 0, &mystate);
	curand_init(1017, blockIdx.x, 0, &mystate2);

	unsigned int j = blockIdx.x;
//for(int j=0; j<1024; ++j) {
//	double inner = 0.0;
//	for(int i=0; i<32; ++i)	{
//		inner += x[i] * sqrt(0.002)*curand_normal_double(&mystate);
//	}

	sdata[threadIdx.x] = x[idx] * sqrt(0.002)*curand_normal_double(&mystate);
	
		
		if(threadIdx.x < 16){
			sdata[threadIdx.x] += sdata[threadIdx.x + 16];
			sdata[threadIdx.x] += sdata[threadIdx.x +  8];
			sdata[threadIdx.x] += sdata[threadIdx.x +  4];
			sdata[threadIdx.x] += sdata[threadIdx.x +  2];
			sdata[threadIdx.x] += sdata[threadIdx.x +  1];
		}

	if(threadIdx.x==0)	
		y[j] = sqrt(2./1024.) * cos( sdata[0] + 6.283185307179586*curand_uniform_double(&mystate2) );
//}
}

template __global__ void doRandomStuff<double,  1>(double* x, double* result, unsigned int n, curandState * rngStates, curandState * rngStatesUni);
template __global__ void doRandomStuff<double,  32>(double* x, double* result, unsigned int n, curandState * rngStates, curandState * rngStatesUni);
template __global__ void doRandomStuff<double,  64>(double* x, double* result, unsigned int n, curandState * rngStates, curandState * rngStatesUni);
template __global__ void doRandomStuff<double, 128>(double* x, double* result, unsigned int n, curandState * rngStates, curandState * rngStatesUni);
template __global__ void doRandomStuff<double, 256>(double* x, double* result, unsigned int n, curandState * rngStates, curandState * rngStatesUni);
template __global__ void doRandomStuff<double, 512>(double* x, double* result, unsigned int n, curandState * rngStates, curandState * rngStatesUni);

template __global__ void doRandomStuff<float,  32>(float* x, float* result, unsigned int n, curandState * rngStates, curandState * rngStatesUni);
template __global__ void doRandomStuff<float,  64>(float* x, float* result, unsigned int n, curandState * rngStates, curandState * rngStatesUni);
template __global__ void doRandomStuff<float, 128>(float* x, float* result, unsigned int n, curandState * rngStates, curandState * rngStatesUni);
template __global__ void doRandomStuff<float, 256>(float* x, float* result, unsigned int n, curandState * rngStates, curandState * rngStatesUni);
template __global__ void doRandomStuff<float, 512>(float* x, float* result, unsigned int n, curandState * rngStates, curandState * rngStatesUni);
