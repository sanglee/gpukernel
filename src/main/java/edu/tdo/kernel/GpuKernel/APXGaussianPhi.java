package edu.tdo.kernel.GpuKernel;

public class APXGaussianPhi extends DLLLoader {

	private double gamma;
	private int d;
	private long adr;

	public APXGaussianPhi(double gamma, int d){
		this.gamma = gamma;
		this.d     = d;
		this.adr   = 0;

		init();
	}

	public double getGamma(){
		return gamma;
	}

	public int getAPXDim(){
		return d;
	}

	private long getAdr(){
		return adr;
	}

	private void setAdr(long val){
		adr = val;
	}

	private native void init();

	public native void destroy(); // explicit destruction of native resources..
	public native double[] transform(double[] x);

	/*
	static{
		System.loadLibrary("APXGaussianPhi");
	}
	*/
	
	static {
		loadDLL("APXGaussianPhi", "libAPXGaussianPhi.so");
	}
}

