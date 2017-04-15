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

#ifndef FFTWBENCH_H
#define FFTWBENCH_H

#include <string>

#include "fftbench.h"
#include "fftw_driver.h"

// 1) input parsing
// 2) storage initialization 	(depends on total)
// 3) library initialization	(depends on geometry)
// 4) data generation		(depends on total)
// 5) warmup			()
// 6) timed loop		(depends on loop)
// 7) library unloading		()
// 8) free			()

namespace FFTW

{
class fftwbench : public fftbench {
	public:
		fftwbench(const fftbench_opt& opt);  // 1,2,3,4
		~fftwbench(); // 7,8
		double run();  // 5,6
	private:
		fftw_driver *_drv;
		int _N;
};

}

#endif

