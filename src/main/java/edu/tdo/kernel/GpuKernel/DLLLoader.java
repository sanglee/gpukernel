package edu.tdo.kernel.GpuKernel;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.net.URL;
import java.net.URLDecoder;
import java.security.CodeSource;
import java.security.ProtectionDomain;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

import org.omg.CORBA_2_3.portable.OutputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DLLLoader{

	static Logger log = LoggerFactory.getLogger( DLLLoader.class );
	
	protected static void loadDLL(String clName, String dllName){
		String libName = null;
		String className = "edu.tdo.kernel.GpuKernel." + clName;
		try{
			log.info("Loading class {}", className );
			Class<?> clazz = Class.forName(className);
			ProtectionDomain pd = clazz.getProtectionDomain();
			log.info("Protection domain is: {}", pd );
			CodeSource cs       = pd.getCodeSource();
			//URL urlTopJAR		= cs.getLocation();
			//String sUrlTopJAR	= URLDecoder.decode(urlTopJAR.getFile(), "UTF-8");
			
			//File fileJAR = new File(sUrlTopJAR);
			//JarFile jf	 = new JarFile(fileJAR);

			//log.info("urltopjar = {}, dllName = {}", sUrlTopJAR, dllName);
			log.info("Looking up dll as resource: {}", dllName );
			URL libUrl = DLLLoader.class.getResource( "/" + dllName );
			log.info("libUrl = {}", libUrl);
			
			if( libUrl != null ){
				
				File fileTmp = File.createTempFile("lib", null);
				log.info( "Extracting to {}", fileTmp);
				
				FileOutputStream out = new FileOutputStream( fileTmp );
				
				InputStream in = libUrl.openStream();
				byte[] buf = new byte[1024];
				int read = in.read( buf );
				while( read > 0 ){
					out.write( buf, 0, read );
					read = in.read( buf );
				}
				out.close();
				in.close();
				
				libName = fileTmp.getCanonicalPath();
				log.info( "Extracted library to {}", libName );
			}

			/*
			 * 
			JarEntry je  = jf.getJarEntry(dllName);
			String s     = je.getName().toLowerCase();

			if(s.equals(dllName)){
				byte[] bytes = getJarBytes(je, jf);

				File fileTmp = File.createTempFile("lib", null);
				log.info( "Extracting {} to {}", je.getName(), fileTmp);
				//fileTmp.deleteOnExit();

				BufferedOutputStream os = new BufferedOutputStream(new FileOutputStream(fileTmp));
				os.write(bytes);
				os.close();

				libName = fileTmp.getCanonicalPath();
			}
			 */
		}catch (ClassNotFoundException e) {
			System.out.println("DLLLoader ERROR: ClassNotFoundException "+className);
		}catch (UnsupportedEncodingException e) {
			System.out.println("DLLLoader ERROR: UnsupportedEncodingException");
		}catch (IOException e) {
			System.out.println("DLLLoader ERROR: IOException");
		}
		
		try{
			log.info("Loading dll {}", libName );
			System.load(libName);
			System.out.println("DLLLoader: LOADED "+dllName);
		}catch (NullPointerException e) {
			System.out.println("DLLLoader ERROR: FILE NOT FOUND: "+dllName);
		}
		
	}

	private static byte[] getJarBytes(JarEntry je, JarFile jf) throws IOException{
		DataInputStream dis = null;
		byte[] a_by = null;
		try {
			long lSize = je.getSize(); 
			if (lSize <= 0  ||  lSize >= Integer.MAX_VALUE) {
				throw new IOException("Invalid size " + lSize + " for entry " + je);
			}
			a_by = new byte[(int)lSize];
			InputStream is = jf.getInputStream(je);
			dis = new DataInputStream(is);
			dis.readFully(a_by);
		} catch (IOException e) {
			throw new IOException();
		} finally {
			if (dis != null) {
				try {
					dis.close();
				} catch (IOException e) {
				}
			}
		}
		return a_by;
	}
}