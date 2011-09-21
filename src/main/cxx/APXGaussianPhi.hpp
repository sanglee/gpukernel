#ifndef __APXGaussianPhi__
#define __APXGaussianPhi__

#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include <curand.h>
#include <curand_kernel.h>

#include "APXGaussianPhi_JNI.h"

template <class T> __global__ void initRNGs(const unsigned int d, const unsigned long seed, curandState * rSN, curandState * rSU);
template <class T, unsigned int B> __global__ void doRandomStuff(T* x, T* y, unsigned int n, const unsigned long seed, curandState *rSN, curandState *rSU);
template <class T> void uploadGPUConfig(const unsigned int d, const T std, const T scale, curandState * rSN, curandState * rSU);

void printCurandState(curandState* s){
	printf("d = %d, v[0] = %d, v[1] = %d, v[2] = %d, v[3] = %d, v[4] = %d\n", s->d, s->v[0], s->v[1], s->v[2], s->v[3], s->v[4]);
	//printf("boxmuller_flag = %d, boxmuller_extra = %f, boxmuller_extra_double = %f\n", s->boxmuller_flag, s->boxmuller_extra, s->boxmuller_extra_double);
}

template <class T>
class APXGaussianPhi {

	private:
		T gamma;
		T std;
		T scale;

		unsigned int d;

	    curandState *rSN;
	    curandState *rSU;

	    curandState *h_rSN;
	    curandState *h_rSU;
	    
		T *h_storage;
		T *d_storage;

		const unsigned int _blocks;
		const unsigned int _threads;

	public:
		APXGaussianPhi (const T gamma, const unsigned int d, const unsigned int blocks = 256, const unsigned int threads = 32) : _blocks(blocks), _threads(threads){
			
			cudaSetDevice(0);
			cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
			
			this->gamma = gamma;
			this->d     = d;
			
			this->std = sqrt(2.0*gamma);
			this->scale = sqrt(2./d);

			h_storage = (T*) malloc(d*sizeof(T));
			cudaMalloc((void **)&d_storage, d*sizeof(T));
			cudaMemset(d_storage, 0, d*sizeof(T));

			dim3 grid(_blocks);
			dim3 block(_threads);
			
			cudaMalloc((void **)&rSN, _threads*d*sizeof(curandState));
			cudaMalloc((void **)&rSU, _threads*d*sizeof(curandState));
			
			cudaMemset(rSN, 0, _threads*d*sizeof(curandState));
			cudaMemset(rSU, 0, _threads*d*sizeof(curandState));

			initRNGs<T><<<grid, block>>>(d, time(0), rSN, rSU);

			//printf("=== init ===\n");
			
			//h_rSN = (curandState*) malloc(_threads*d*sizeof(curandState));
			//cudaMemcpy(h_rSN, rSN, _threads*d*sizeof(curandState), cudaMemcpyDeviceToHost);
			
			//h_rSU = (curandState*) malloc(_threads*d*sizeof(curandState));
			//cudaMemcpy(h_rSU, rSU, _threads*d*sizeof(curandState), cudaMemcpyDeviceToHost);
			
			//printCurandState(&h_rSN[10*_threads]);

			//printf("rSN adress = %p\n", rSN);
			//printf("rSU adress = %p\n", rSU);
			
			uploadGPUConfig<T>(d, std, scale, rSN, rSU);
		}

		~APXGaussianPhi(){
			free(h_storage);
			cudaFree(d_storage);
			cudaFree(rSN);
			cudaFree(rSU);
		}

		T* transform(T* x, const unsigned int n){

			cudaSetDevice(0);

			T* d_x;
			cudaMalloc((void**)&d_x, n*sizeof(T));

			dim3 grid(_blocks);
			dim3 block(_threads);

			//initRNGs<T><<<grid, block>>>(d, clock(), rSN, rSU);
			//cudaMemcpy(rSN, h_rSN, _threads*d*sizeof(curandState), cudaMemcpyHostToDevice);
			//cudaMemcpy(rSU, h_rSU, _threads*d*sizeof(curandState), cudaMemcpyHostToDevice);
			//printf("=== before transform ===\n");
			//cudaMemcpy(h_rSN, rSN, _threads*d*sizeof(curandState), cudaMemcpyDeviceToHost);
			//printCurandState(&h_rSN[10*_threads]);
			
			//printf("rSN adress = %p\n", rSN);
			//printf("rSU adress = %p\n", rSU);

			cudaMemcpy(d_x, x, n*sizeof(T), cudaMemcpyHostToDevice);
			switch(_threads){
				case   1: {doRandomStuff<T,   1><<<grid, block>>>(d_x, d_storage, n, time(0), rSN, rSU); break; }
				case  32: {doRandomStuff<T,  32><<<grid, block>>>(d_x, d_storage, n, time(0), rSN, rSU); break; }
				case  64: {doRandomStuff<T,  64><<<grid, block>>>(d_x, d_storage, n, time(0), rSN, rSU); break; }
				case 128: {doRandomStuff<T, 128><<<grid, block>>>(d_x, d_storage, n, time(0), rSN, rSU); break; }
				case 256: {doRandomStuff<T, 256><<<grid, block>>>(d_x, d_storage, n, time(0), rSN, rSU); break; }
				default:  {doRandomStuff<T, 512><<<grid, block>>>(d_x, d_storage, n, time(0), rSN, rSU); break; }
			}
			cudaFree(d_x);
			cudaMemcpy(h_storage, d_storage, d*sizeof(T), cudaMemcpyDeviceToHost);

			//printf("=== after transform ===\n");
			//cudaMemcpy(h_rSN, rSN, _threads*d*sizeof(curandState), cudaMemcpyDeviceToHost);
			//printCurandState(h_rSN);

			return h_storage;
		}

		T getGamma(){
			return gamma;
		}

		unsigned int getAPXDim(){
			return d;
		}
};

JNIEXPORT void JNICALL Java_edu_tdo_kernel_GpuKernel_APXGaussianPhi_init(JNIEnv *jvm, jobject j_this){

	jclass JAPXGaussianPhi = jvm->GetObjectClass(j_this);

	jmethodID getGamma  = jvm->GetMethodID(JAPXGaussianPhi, "getGamma",  "()D");
	jmethodID getAPXDim = jvm->GetMethodID(JAPXGaussianPhi, "getAPXDim", "()I");
	jmethodID getAdr    = jvm->GetMethodID(JAPXGaussianPhi, "getAdr",    "()J");
	jmethodID setAdr    = jvm->GetMethodID(JAPXGaussianPhi, "setAdr",    "(J)V");

	double        gamma = jvm->CallDoubleMethod(j_this, getGamma);
	unsigned int      d = jvm->CallIntMethod(j_this, getAPXDim);
	long            adr = jvm->CallLongMethod(j_this, getAdr);

	if(adr) delete (APXGaussianPhi<double>*)adr;
	APXGaussianPhi<double>* phi = new APXGaussianPhi<double>(gamma, d);

	jvm->CallVoidMethod(j_this, setAdr, (long)phi);
}

JNIEXPORT void JNICALL Java_edu_tdo_kernel_GpuKernel_APXGaussianPhi_destroy(JNIEnv *jvm, jobject j_this){

	jclass JAPXGaussianPhi = jvm->GetObjectClass(j_this);

	jmethodID getAdr = jvm->GetMethodID(JAPXGaussianPhi, "getAdr", "()J");
	jmethodID setAdr = jvm->GetMethodID(JAPXGaussianPhi, "setAdr", "(J)V");

	long adr = jvm->CallLongMethod(j_this, getAdr);

	if(adr){
		APXGaussianPhi<double> *phi = (APXGaussianPhi<double>*) adr;
		delete phi;
		jvm->CallVoidMethod(j_this, setAdr, 0);
	}
}

JNIEXPORT jdoubleArray JNICALL Java_edu_tdo_kernel_GpuKernel_APXGaussianPhi_transform(JNIEnv *jvm, jobject j_this, jdoubleArray j_x){

	jclass JAPXGaussianPhi = jvm->GetObjectClass(j_this);

	jmethodID getAdr = jvm->GetMethodID(JAPXGaussianPhi, "getAdr", "()J");
	long adr         = jvm->CallLongMethod(j_this, getAdr);
	
	APXGaussianPhi<double> *phi = (APXGaussianPhi<double>*) adr;
	
	int n            = jvm->GetArrayLength(j_x);

	double *x        = jvm->GetDoubleArrayElements(j_x, JNI_FALSE);
	double *y        = phi->transform(x, n);

	jdoubleArray j_y = jvm->NewDoubleArray(phi->getAPXDim());
	jvm->SetDoubleArrayRegion(j_y, 0, phi->getAPXDim(), y);
	
	jvm->ReleaseDoubleArrayElements(j_x, x, JNI_FALSE);

	return j_y;
}
#endif
