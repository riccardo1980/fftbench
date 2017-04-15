/*
* Authors: 
*   Riccardo Zanella
*    Dept. of Mathematics, University of Ferrara, Italy
*    riccardo.zanella@unife.it
*
* Copyright (C) 2011 by R. Zanella
* ------------------------------------------------------------------------------
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
* ==============================================================================
*/

#include <iostream>
#include <string>
#include <sstream>

#include <cstdlib>

#include <complex>
#include <omp.h>

#include "mklbench.h"
#include "R2C_fft_driver.h"
//#include "D2Z_fft_driver.h"
#include "C2C_fft_driver.h"
#include "Z2Z_fft_driver.h"

using std::complex;
using std::string;
using std::cerr;
using std::endl;

namespace MKL {

// 1) input parsing
// 2) storage initialization 	(depends on total)
// 3) library initialization	(depends on geometry)
// 4) data generation		(depends on total)
// 5) warmup			()
// 6) timed loop		(depends on loop)
// 7) library unloading		()
// 8) free			()

mklbench::mklbench(const fftbench_opt& opt) 
	: _drv(NULL), _N(opt.loops) {

		switch (opt.type){
			case FFT_R2C:
				_drv = new R2C_fft_driver(opt); break;
//			case FFT_D2Z:
//				_drv = new D2Z_fft_driver(opt); break;
			case FFT_C2C:
				_drv = new C2C_fft_driver(opt); break;
			case FFT_Z2Z:
				_drv = new Z2Z_fft_driver(opt); break;
			default:
				cerr << "Can't initialize driver for this kind of transform!" << endl;
	}

	_drv->randfill();
	_drv->initBackend();
		
}

mklbench::~mklbench(){

	_drv->finalizeBackend();
	delete _drv;

}

double mklbench::run(){
int ii;
double start, stop;

	// warmup
	_drv->run();

	start = omp_get_wtime();
	for (ii = 0; ii < _N; ii++)
		_drv->run();

	stop = omp_get_wtime();


	return ( (stop - start ) / (double) _N * 1000.0); 

}

}

