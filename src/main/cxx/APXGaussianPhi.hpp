#ifndef __APXGaussianPhi__
#define __APXGaussianPhi__

#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include <curand.h>
#include <curand_kernel.h>

#include "APXGaussianPhi_JNI.h"

__global__ void initRNG(curandState * const rngStates, const unsigned int d, const unsigned int seed);
template <class T, unsigned int B> __global__ void doRandomStuff(T* x, T* y, unsigned int n, curandState * rngStates, curandState * rngStatesUni);
template <class T> void uploadGPUConfig(const unsigned int d, const T std, const T scale, curandState * const rngStates, curandState * const rngStatesUni);

template <class T>
class APXGaussianPhi {

	private:
		T gamma;
		T std;
		T scale;

		unsigned int d;

	    curandState *rngStates;
	    curandState *rngStatesUni;

		T* h_storage;
		T* d_storage;

		const unsigned int _blocks;
		const unsigned int _threads;

	public:
		APXGaussianPhi (const T gamma, const unsigned int d, const unsigned int blocks = 1024, const unsigned int threads = 32) : _blocks(blocks), _threads(threads){

			printf("version = 1.3\n");
			
			this->gamma = gamma;
			this->d = d;
			
			this->std = sqrt(2.0*gamma);
			this->scale = sqrt(2./d);
			

			h_storage =(T*) malloc(d*sizeof(T));
			cudaMalloc((void**)&d_storage, d*sizeof(T));

			dim3 igrid(256);
			dim3 block(threads);

			cudaMalloc((void **)&rngStates, d*threads*sizeof(curandState));
			cudaMalloc((void **)&rngStatesUni, d*sizeof(curandState));
			initRNG<<<igrid, block>>>(rngStates, d*threads, time(0));
			initRNG<<<igrid, block>>>(rngStatesUni, d, time(0)*2);
			
			uploadGPUConfig<T>(d, std, scale, rngStates, rngStatesUni);
		}

		~APXGaussianPhi(){
			free(h_storage);
			cudaFree(d_storage);
			cudaFree(rngStates);
		}

		T* transform(T* x, const unsigned int n){

			T* d_x;
			cudaMalloc((void**)&d_x, n*sizeof(T));

			dim3 grid(_blocks);
			dim3 block(_threads);

			cudaMemcpy(d_x, x, n*sizeof(T), cudaMemcpyHostToDevice);
			switch(_threads){
				case  1: {doRandomStuff<T,  1><<<grid, block>>>(d_x, d_storage, n, rngStates, rngStatesUni); break; }
				/*
				case  64: {doRandomStuff<T,  64><<<grid, block>>>(d_x, d_storage, n, rngStates, rngStatesUni); break; }
				case 128: {doRandomStuff<T, 128><<<grid, block>>>(d_x, d_storage, n, rngStates, rngStatesUni); break; }
				case 256: {doRandomStuff<T, 256><<<grid, block>>>(d_x, d_storage, n, rngStates, rngStatesUni); break; }
				default:  {doRandomStuff<T, 512><<<grid, block>>>(d_x, d_storage, n, rngStates, rngStatesUni); break; }
				*/
			}
			cudaFree(d_x);
			cudaMemcpy(h_storage, d_storage, d*sizeof(T), cudaMemcpyDeviceToHost);

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
	
	//jmethodID getAPXDim = jvm->GetMethodID(JAPXGaussianPhi, "getAPXDim", "()I");
	//unsigned int      d = jvm->CallIntMethod(j_this, getAPXDim);
	
	APXGaussianPhi<double> *phi = (APXGaussianPhi<double>*) adr;
	//int d = phi->getAPXDim();
	
	int n            = jvm->GetArrayLength(j_x);

	double *x        = jvm->GetDoubleArrayElements(j_x, JNI_FALSE);
	double *y        = phi->transform(x, n);

	jdoubleArray j_y = jvm->NewDoubleArray(phi->getAPXDim());
	jvm->SetDoubleArrayRegion(j_y, 0, phi->getAPXDim(), y);
	
	/*
	printf("Input = \n");
	for(int i=0; i<n; ++i) {
		printf("%3d: %3.8e\n", i, x[i]); 
	}
	printf("\n\n\n");
		
	printf("Output = \n (%d) dim", d);
	for(int i=0; i<d; ++i) {
		printf("%3d: %3.8e\n", i, y[i]); 
	}
	printf("\n\n\n");
	fflush(stdout);
	*/
	
	jvm->ReleaseDoubleArrayElements(j_x, x, JNI_FALSE);

	return j_y;
}
#endif
