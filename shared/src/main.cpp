/*
* Authors: 
*   Riccardo Zanella
*    Dept. of Mathematics, University of Ferrara, Italy
*    riccardo.zanella@unife.it
*
* Copyright (C) 2011 by R. Zanella
* --------------------------------------------------------------------------
* COPYRIGHT NOTIFICATION
*
* Permission to copy and modify this software and its documentation for 
* internal research use is granted, provided that this notice is retained 
* thereon and on all copies or modifications. The authors and their
* respective Universities makes no representations as to the suitability 
* and operability of this software for any purpose. It is provided "as is"
* without express or implied warranty. Use of this software for commercial
* purposes is expressly prohibited without contacting the authors.
*
* This program is free software; you can redistribute it and/or modify it
* under the terms of the GNU General Public License as published by the
* Free Software Foundation; either version 3 of the License, or (at your 
* option) any later version.
*
* This program is distributed in the hope that it will be useful, but 
* WITHOUT ANY WARRANTY; without even the implied warranty of 
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
* See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along 
* with this program; if not, either visite http://www.gnu.org/licenses/
* or write to
* Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
* ==========================================================================
*/

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <algorithm>
#include <complex>
#include <cstdlib>

#include "fftbench.h"

#ifdef ENABLEFFTW
#include "FFTW/fftwbench.h"
#endif

#ifdef ENABLECUFFT
#include "CUFFT/cufftbench.h"
#endif

#ifdef ENABLEMKL
#include "MKL/mklbench.h"
#endif

#if defined (ENABLEFFTW) || defined (ENABLEMKL)
#define ENABLETHREADS
#endif


using std::string;
using std::cout;
using std::cerr;
using std::endl;

void print_help (const char *me){
	cout 	<< "Usage: " << me << " [options]" << endl << endl
				
		<< "-h      prints this help" << endl << endl

		<< "-l      loops" << endl
		<< "        default: 100" << endl << endl
#ifdef ENABLTHREADS
		<< "-n      threads (CPU implementation)" << endl
		<< "        default: 1" << endl << endl
#endif
		
		<< "-p      provider (one of the following)" << endl
#ifdef ENABLEFFTW
		<< "        FFTW (multithreading)" << endl
#endif
#ifdef ENABLECUFFT
		<< "        CUFFT implementation" << endl
#endif
#ifdef ENABLEMKL
		<< "        MKL implementation" << endl
#endif
		<< endl

		<< "-s      size (string, es: 256x128x...)" << endl
		<< "        default: 256x256" << endl << endl  

		<< "-t      type (one of the following)" << endl
		<< "        R2C single precision real to complex" << endl
		<< "        D2Z double precision real to complex" << endl
		<< "        C2C single precision complex to complex" << endl
		<< "        Z2Z double precision complex to complex" << endl
		<< "        default: D2Z" << endl << endl

		<< "-v      verbosity level [0,1]" << endl
		<< "        default: 0" << endl << endl;

}
template <typename T>
T stringToNum ( const string &Text ){ 
	std::istringstream iss(Text);
	T result;
	return iss >> result ? result : 0;
}

typedef enum { PROV_UNSET, PROV_FFTW, PROV_CUFFT, PROV_MKL } provType;

int main (int argc, char *argv[]){
string type_s, prov_s;
int verbose;

provType prov;

// set default options
fftbench_opt opt;
	opt.loops = 100;
	opt.type = FFT_D2Z;
	opt.size = string("256x256");
	opt.nthreads = 1;

	type_s = string("D2Z");
	prov_s = string("FFTW");
	verbose = 0;

// input parsing
	for (int ii = 1; ii < argc; ii++){
		if (argv[ii][0] != '-')
			break;
		char *t = &argv[ii][1];
		switch (*t){


			case 'h':
				print_help(argv[0]); exit(EXIT_SUCCESS);
				break;

			case 'l': 
				if (*(t+1) != '\0')
					opt.loops = stringToNum<int>(string(t+1)); 
				else{
					ii++;
					if ( ii != argc && argv[ii][0] != '-')
						opt.loops = stringToNum<int>(string(argv[ii]));
					else{
						cerr << "Flag -" << t 
							<< " requires a parameter" << endl; 
						exit(EXIT_FAILURE);
					}
				}
				break;

			case 'n':
				if (*(t+1) != '\0')
					opt.nthreads = stringToNum<int>(string(t+1)); 
				else{
					ii++;
					if ( ii != argc && argv[ii][0] != '-')
						opt.nthreads = stringToNum<int>(string(argv[ii]));
					else{
						cerr << "Flag -" << t 
							<< " requires a parameter" << endl; 
						exit(EXIT_FAILURE);
					}
				}
				break;	

			case 'p':
				if (*(t+1) != '\0')
					prov_s = string(t+1); 
				else{
					ii++;
					if ( ii != argc && argv[ii][0] != '-')
						prov_s = string(argv[ii]);
					else{
						cerr << "Flag -" << t 
							<< " requires a parameter" << endl; 
						exit(EXIT_FAILURE);
					}
				}
				break;	

			case 's':
				if (*(t+1) != '\0')
					opt.size = string(t+1); 
				else{
					ii++;
					if ( ii != argc && argv[ii][0] != '-')
						opt.size = string(argv[ii]);
					else{
						cerr << "Flag -" << t 
							<< " requires a parameter" << endl; 
						exit(EXIT_FAILURE);
					}
				}
				break;

			case 't':
				if (*(t+1) != '\0')
					type_s = string(t+1); 
				else{
					ii++;
					if ( ii != argc && argv[ii][0] != '-')
						type_s = string(argv[ii]);
					else{
						cerr << "Flag -" << t 
							<< " requires a parameter" << endl; 
						exit(EXIT_FAILURE);
					}
				}
				break;

			case 'v':
				if (*(t+1) != '\0')
					verbose = stringToNum<int>(string(t+1)); 
				else{
					ii++;
					if ( ii != argc && argv[ii][0] != '-')
						verbose = stringToNum<int>(string(argv[ii]));
					else{
						cerr << "Flag -" << t 
							<< " requires a parameter" << endl; 
						exit(EXIT_FAILURE);
					}
				}
				break;

				
			default:
				cout << "unrecognized flag " << *t << endl;
				exit(EXIT_FAILURE);
		}

	}

	// parameters checking
	if ( opt.loops < 1 ){
		cerr << "Error on loop number" << endl; 
		exit(EXIT_FAILURE);
	}
	// FFT_R2C, FFT_D2Z, FFT_C2C, Z2Z
	opt.type = FFT_UNSET;
	if ( !type_s.compare("R2C") )
		opt.type = FFT_R2C;
	if ( !type_s.compare("D2Z") )
		opt.type = FFT_D2Z;
	if ( !type_s.compare("C2C") )
		opt.type = FFT_C2C;
	if ( !type_s.compare("Z2Z") )
		opt.type = FFT_Z2Z;
	if (opt.type == FFT_UNSET){
		cerr << "unrecognized fft type: " 
			<< type_s << endl; 
		exit(EXIT_FAILURE);
	}

	prov = PROV_UNSET;
	if ( !prov_s.compare("FFTW") )
		prov = PROV_FFTW;
	if ( !prov_s.compare("CUFFT") )
		prov = PROV_CUFFT;
	if ( !prov_s.compare("MKL") )
		prov = PROV_MKL;
if ( prov == PROV_UNSET){
		cerr << "unrecognized provider type: " 
			<< prov_s << endl; 
		exit(EXIT_FAILURE);
	}
	if ( opt.nthreads < 1 ){
		cerr << "Error on thread number" << endl; 
		exit(EXIT_FAILURE);
	}
	if ( verbose < 0 || verbose > 1 ){
		cerr << "Error on verbosity level" << endl; 
		exit(EXIT_FAILURE);
	}

	if (verbose){
		cout << "loops:        " << opt.loops << endl 
		     << "type:         " << type_s << endl
		     << "size:         " << opt.size << endl
		     << "provider:     " << prov_s << endl;
		     if (prov == PROV_FFTW || prov == PROV_MKL)
			     cout << "nthreads:     " << opt.nthreads << endl;
		     cout << endl;    
	}


	fftbench *bench;
  switch (prov){
#ifdef ENABLEFFTW
    case PROV_FFTW:	
      bench = new FFTW::fftwbench(opt); break;
#endif
#ifdef ENABLECUFFT
    case PROV_CUFFT:
      bench = new CUFFT::cufftbench(opt); break;
#endif
#ifdef ENABLEMKL
    case PROV_MKL:
      bench = new MKL::mklbench(opt); break;
#endif
    default:
      cerr<<"I shouldn't be here..." << endl;
      exit(EXIT_FAILURE);
  }

	// BENCHMARK
	double elapsedTime = bench->run();
	printf("Elapsed time %e [ms]\n", elapsedTime);
	delete bench;

	return 0;
}
