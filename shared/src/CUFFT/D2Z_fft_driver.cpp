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

#include <string>
#include <iostream>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

#include "parseUtils.h"
#include "cufftbench.h"
#include "D2Z_fft_driver.h"
using std::string;

namespace CUFFT {

D2Z_fft_driver::D2Z_fft_driver(const fftbench_opt& opt) 
: _opt(opt), _in(NULL), _out(NULL), _IN(NULL),
	_Naxes(0), _axesDim(NULL)
{
	charToBlocks(&_axesDim, &_Naxes, opt.size);			
	_Ntot = _axesDim[_Naxes-1];
	_Ncomplex = _axesDim[_Naxes-1] / 2 + 1;

	for(int ii = 0; ii < _Naxes-1; ii++){
		_Ntot *= _axesDim[ii];
		_Ncomplex *= _axesDim[ii];
	}
	_in = reinterpret_cast<double*>
		(malloc (_Ntot * sizeof(double)));
	_out = reinterpret_cast<double*>
		(malloc (_Ntot * sizeof(double)));
	
	CUDA_SAFE_CALL(	cudaMalloc( (void**) &g_in , 
			sizeof(double)*_Ntot) );

	CUDA_SAFE_CALL( cudaMalloc( (void**) &g_out, 
			sizeof(double)*_Ntot) );

	CUDA_SAFE_CALL( cudaMalloc( (void**) &_IN, 
			sizeof(cufftDoubleComplex)*_Ncomplex) );

}

D2Z_fft_driver::~D2Z_fft_driver(){
	free(_axesDim);
	free(_in);
	free(_out);
	CUDA_SAFE_CALL( cudaFree(g_in));
	CUDA_SAFE_CALL( cudaFree(g_out));
	CUDA_SAFE_CALL( cudaFree(_IN));
}
void D2Z_fft_driver::randfill(){
	for(int ii = 0;  ii < _Ntot; ii++)
		_in[ii] = rand()/((double) RAND_MAX);

	CUDA_SAFE_CALL(
	cudaMemcpy(g_in, _in, sizeof(double)*_Ntot, 
			cudaMemcpyHostToDevice)
	);
}
void D2Z_fft_driver::initBackend(){
	switch(_Naxes){
		case 1: 
			CUFFT_SAFE_CALL( 
				cufftPlan1d( &_r2c, _axesDim[0],
				CUFFT_D2Z, 1));
			CUFFT_SAFE_CALL(
				cufftPlan1d( &_c2r, _axesDim[0],
				CUFFT_Z2D, 1));
			break;
		case 2:
			CUFFT_SAFE_CALL(
				cufftPlan2d( &_r2c, _axesDim[0],
				_axesDim[1], CUFFT_D2Z));
			CUFFT_SAFE_CALL(
				cufftPlan2d( &_c2r, _axesDim[0],
				_axesDim[1], CUFFT_Z2D));
			break;
		case 3:
			CUFFT_SAFE_CALL(
				cufftPlan3d( &_r2c, _axesDim[0],
				_axesDim[1], _axesDim[2], CUFFT_D2Z));
			CUFFT_SAFE_CALL(
				cufftPlan3d( &_c2r, _axesDim[0],
				_axesDim[1], _axesDim[2], CUFFT_Z2D));
			break;
		default:
			std::cerr << "Can handle only 1 to 3 Dim signals!"
				<< std::endl;
			exit(EXIT_FAILURE);
	}
}
void D2Z_fft_driver::run(){
	CUFFT_SAFE_CALL(
	cufftExecD2Z(_r2c, 
		reinterpret_cast< cufftDoubleReal * >(g_in),
		_IN) 
	);
	CUFFT_SAFE_CALL(
	cufftExecZ2D(_c2r, _IN,
		reinterpret_cast< cufftDoubleReal * >(g_out))
	);
}
void D2Z_fft_driver::finalizeBackend(){
	CUFFT_SAFE_CALL( cufftDestroy(_c2r) );
	CUFFT_SAFE_CALL( cufftDestroy(_r2c) );
}


}

