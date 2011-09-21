#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>

__constant__ __device__ long long c_conf[1024]; // 8 KiB ConstMem

#define TWOPI 6.283185307179586

template <class T>
struct GPUConfig{

	unsigned int d;

	T std;
	T scale;

	curandState * rSN;
	curandState * rSU;
};

template <>
struct GPUConfig<double>{

	unsigned int d;

	double std;
	double scale;

	curandState * rSN;
	curandState * rSU;
};

template <class T>
__global__ void initRNGs(const unsigned int d, const unsigned long seed, curandState * rSN, curandState * rSU){

	const unsigned int gridSize = d * blockDim.x;

	for(unsigned int i = blockIdx.x; i < d; i += gridDim.x){
		unsigned int tid = i * blockDim.x + threadIdx.x;
		
		curandState n_state = rSN[tid];
		curandState u_state = rSU[i];
		
		curand_init(seed, tid, 0, &n_state);
		rSN[tid] = n_state;
		
		//if(threadIdx.x == 0){
		//	curand_init(seed, gridSize+tid, 0, &u_state);
		//	rSU[tid] = u_state;
		//}
    }
}

template __global__ void initRNGs<double>(const unsigned int d, const unsigned long seed, curandState *rSN, curandState *rSU);
template __global__ void initRNGs<float>(const unsigned int d, const unsigned long seed, curandState *rSN, curandState *rSU);

template <class T>
void uploadGPUConfig(const unsigned int d, const T std, const T scale, curandState *rSN, curandState *rSU){

	printf("BUILD %s\n", __TIMESTAMP__);
	fflush(stdout);

	GPUConfig<T> *gc = (GPUConfig<T>*) malloc(sizeof(GPUConfig<T>));

	gc->d		= d;
	gc->std		= std;
	gc->scale	= scale;
	gc->rSN		= rSN;
	gc->rSU		= rSU;

	cudaMemcpyToSymbol(c_conf, gc, sizeof(GPUConfig<T>));

	free(gc);
}

template void uploadGPUConfig<double>(const unsigned int d, const double std, const double scale, curandState *rSN, curandState *rSU);
template void uploadGPUConfig<float>(const unsigned int d, const float std, const float scale, curandState *rSN, curandState *rSU);

template <class T, unsigned int B>
__global__ void doRandomStuff(T* x, T* y, const unsigned int n, const unsigned long seed, curandState *rSN, curandState *rSU){

	const GPUConfig<T>* conf = (GPUConfig<T>*)&c_conf;

	const T std    = conf->std;
	const T scale  = conf->scale;

	const unsigned int d = conf->d;
	const unsigned int gridSize = d * blockDim.x;

	volatile __shared__ T sdata[B];

	for(unsigned int i = blockIdx.x; i < d/2; i += gridDim.x){

		unsigned int tid = i * blockDim.x + threadIdx.x;
		
		// load RNG states
		curandState n_state = rSN[tid];

		sdata[threadIdx.x] = (T)0.0;
	
		for(unsigned int idx = threadIdx.x; idx < n; idx += blockDim.x){
			sdata[threadIdx.x] += x[idx] * (std * curand_normal_double(&n_state));
		}
		if(B>1){
			if(B>32) __syncthreads();
		
			if(B>256){
				if(threadIdx.x < 256)
					sdata[threadIdx.x] += sdata[threadIdx.x + 256];
				__syncthreads();
			}
			if(B>128){
				if(threadIdx.x < 128)
					sdata[threadIdx.x] += sdata[threadIdx.x + 128];
				__syncthreads();
			}
			if(B>64){
				if(threadIdx.x < 64)
					sdata[threadIdx.x] += sdata[threadIdx.x + 64];
				__syncthreads();
			}
			if(B>32){
				if(threadIdx.x < 32)
					sdata[threadIdx.x] += sdata[threadIdx.x + 32];
			}
			if(threadIdx.x < 16){
				sdata[threadIdx.x] += sdata[threadIdx.x + 16];
				sdata[threadIdx.x] += sdata[threadIdx.x +  8];
				sdata[threadIdx.x] += sdata[threadIdx.x +  4];
				sdata[threadIdx.x] += sdata[threadIdx.x +  2];
				sdata[threadIdx.x] += sdata[threadIdx.x +  1];
			}
		}
		// write back
		rSN[tid] = n_state;

		if(threadIdx.x == 0){
			y[i] = scale * cos(sdata[0]);
			
		}
	}
	
	for(unsigned int i = (d/2)+blockIdx.x; i < d; i += gridDim.x){

		unsigned int tid = i * blockDim.x + threadIdx.x;
		
		// load RNG state
		curandState n_state = rSN[tid];

		sdata[threadIdx.x] = (T)0.0;
	
		for(unsigned int idx = threadIdx.x; idx < n; idx += blockDim.x){
			sdata[threadIdx.x] += x[idx] * (std * curand_normal_double(&n_state));
		}
		if(B>1){
			if(B>32) __syncthreads();
		
			if(B>256){
				if(threadIdx.x < 256)
					sdata[threadIdx.x] += sdata[threadIdx.x + 256];
				__syncthreads();
			}
			if(B>128){
				if(threadIdx.x < 128)
					sdata[threadIdx.x] += sdata[threadIdx.x + 128];
				__syncthreads();
			}
			if(B>64){
				if(threadIdx.x < 64)
					sdata[threadIdx.x] += sdata[threadIdx.x + 64];
				__syncthreads();
			}
			if(B>32){
				if(threadIdx.x < 32)
					sdata[threadIdx.x] += sdata[threadIdx.x + 32];
			}
			if(threadIdx.x < 16){
				sdata[threadIdx.x] += sdata[threadIdx.x + 16];
				sdata[threadIdx.x] += sdata[threadIdx.x +  8];
				sdata[threadIdx.x] += sdata[threadIdx.x +  4];
				sdata[threadIdx.x] += sdata[threadIdx.x +  2];
				sdata[threadIdx.x] += sdata[threadIdx.x +  1];
			}
		}
		// write back
		rSN[tid] = n_state;
			
		if(threadIdx.x == 0){
			y[i] = scale * sin(sdata[0]);
		}
	}
}

/*
template <class T, unsigned int B>
__global__ void doRandomStuff(T* x, T* y, const unsigned int n, const unsigned long seed, curandState *rSN, curandState *rSU){

	const GPUConfig<T>* conf = (GPUConfig<T>*)&c_conf;

	const T std    = conf->std;
	const T scale  = conf->scale;

	const unsigned int d = conf->d;
	const unsigned int gridSize = d * blockDim.x;

	volatile __shared__ T sdata[B];

	for(unsigned int i = blockIdx.x; i < d; i += gridDim.x){

		unsigned int tid = i * blockDim.x + threadIdx.x;
		
		// load RNG states
		curandState n_state = rSN[tid];
		curandState u_state = rSU[tid];
		//if(threadIdx.x == 0) u_state = rSU[i];

		sdata[threadIdx.x] = (T)0.0;
	
		for(unsigned int idx = threadIdx.x; idx < n; idx += blockDim.x){
			sdata[threadIdx.x] += x[idx] * (std * curand_normal_double(&n_state));
		}
		if(B>1){
		if(B>32) __syncthreads();
		
		if(B>256){
			if(threadIdx.x < 256)
				sdata[threadIdx.x] += sdata[threadIdx.x + 256];
			__syncthreads();
		}
		if(B>128){
			if(threadIdx.x < 128)
				sdata[threadIdx.x] += sdata[threadIdx.x + 128];
			__syncthreads();
		}
		if(B>64){
			if(threadIdx.x < 64)
				sdata[threadIdx.x] += sdata[threadIdx.x + 64];
			__syncthreads();
		}
		if(B>32){
			if(threadIdx.x < 32)
				sdata[threadIdx.x] += sdata[threadIdx.x + 32];
		}
		if(threadIdx.x < 16){
			sdata[threadIdx.x] += sdata[threadIdx.x + 16];
			sdata[threadIdx.x] += sdata[threadIdx.x +  8];
			sdata[threadIdx.x] += sdata[threadIdx.x +  4];
			sdata[threadIdx.x] += sdata[threadIdx.x +  2];
			sdata[threadIdx.x] += sdata[threadIdx.x +  1];
		}
		}
		// write back
		rSN[tid] = n_state;
			
		if(threadIdx.x == 0){
			y[i] = scale * cos(sdata[0] + 6.283185307179586*curand_uniform_double(&u_state));
			

		}
					rSU[tid] = u_state;
	}
}
*/

template __global__ void doRandomStuff<double,   1>(double* x, double* result, const unsigned int n, const unsigned long seed, curandState *rSN, curandState *rSU);
template __global__ void doRandomStuff<double,  32>(double* x, double* result, const unsigned int n, const unsigned long seed, curandState *rSN, curandState *rSU);
template __global__ void doRandomStuff<double,  64>(double* x, double* result, const unsigned int n, const unsigned long seed, curandState *rSN, curandState *rSU);
template __global__ void doRandomStuff<double, 128>(double* x, double* result, const unsigned int n, const unsigned long seed, curandState *rSN, curandState *rSU);
template __global__ void doRandomStuff<double, 256>(double* x, double* result, const unsigned int n, const unsigned long seed, curandState *rSN, curandState *rSU);
template __global__ void doRandomStuff<double, 512>(double* x, double* result, const unsigned int n, const unsigned long seed, curandState *rSN, curandState *rSU);

template __global__ void doRandomStuff<float,   1>(float* x, float* result, const unsigned int n, const unsigned long seed, curandState *rSN, curandState *rSU);
template __global__ void doRandomStuff<float,  32>(float* x, float* result, const unsigned int n, const unsigned long seed, curandState *rSN, curandState *rSU);
template __global__ void doRandomStuff<float,  64>(float* x, float* result, const unsigned int n, const unsigned long seed, curandState *rSN, curandState *rSU);
template __global__ void doRandomStuff<float, 128>(float* x, float* result, const unsigned int n, const unsigned long seed, curandState *rSN, curandState *rSU);
template __global__ void doRandomStuff<float, 256>(float* x, float* result, const unsigned int n, const unsigned long seed, curandState *rSN, curandState *rSU);
template __global__ void doRandomStuff<float, 512>(float* x, float* result, const unsigned int n, const unsigned long seed, curandState *rSN, curandState *rSU);
