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

#ifndef FFTBENCH_H
#define FFTBENCH_H

#include <string>


typedef enum { FFT_UNSET, 
	       FFT_R2C, FFT_D2Z, FFT_C2C, FFT_Z2Z,
               } fft_type;

class fftbench_opt {
	public:
	int loops;
	fft_type type;
	std::string size;
	int nthreads;
	fftbench_opt(){};
	~fftbench_opt(){};
	fftbench_opt(const fftbench_opt &rhs){
		this->loops = rhs.loops; 
		this->type = rhs.type; 
		this->size = rhs.size; 
		this->nthreads = rhs.nthreads; 
	}	
};

// 1) input parsing
// 2) storage initialization 	(depends on total)
// 3) library initialization	(depends on geometry)
// 4) data generation		(depends on total)
// 5) warmup			()
// 6) timed loop		(depends on loop)
// 7) library unloading		()
// 8) free			()


class fftbench{
	public:
		fftbench(){};
		virtual ~fftbench() {}; // 7,8
		virtual double run() = 0;  // 5,6
};


#endif

