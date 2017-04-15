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

#ifndef CUFFT_DRIVER_H
#define CUFFT_DRIVER_H

namespace CUFFT{
// Macro to catch CUDA errors in CUDA runtime calls
#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(call)                                              \
	do {                                                              \
		cudaError_t err = call;                                   \
		if (cudaSuccess != err) {                                 \
			fprintf (stderr,                                  \
			"Cuda error in file '%s' in line %i : %s.",       \
			__FILE__, __LINE__,                               \
			cudaGetErrorString(err) );                        \
			printf("\n");                                     \
			exit(EXIT_FAILURE);                               \
		}                                                         \
	} while (0)
#endif

// Macro to catch CUDA errors in CUDA runtime calls
#ifndef CUFFT_SAFE_CALL
#define CUFFT_SAFE_CALL(call)                                              \
	do {                                                              \
		cufftResult err = call;                                   \
		if (CUFFT_SUCCESS != err) {                                 \
			fprintf (stderr,                                  \
			"Cufft error in file '%s' in line %i : %s.",       \
			__FILE__, __LINE__,  "\n");                           \
			exit(EXIT_FAILURE);                               \
		}                                                         \
	} while (0)
#endif


class cufft_driver {
	public:
		// 2
		cufft_driver(){};

		// 8
		virtual ~cufft_driver(){};

		// 4
		virtual void randfill() = 0;

		// 3 	
		virtual void initBackend() = 0;

		// 6 
		virtual void run() = 0;

		// 7
		virtual void finalizeBackend() = 0;
};

}

#endif

