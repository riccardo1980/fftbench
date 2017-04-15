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
#include "R2C_fft_driver.h"
using std::complex;
using std::string;

namespace CUFFT {

R2C_fft_driver::R2C_fft_driver(const fftbench_opt& opt) 
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
	_in = reinterpret_cast<float*>
		(malloc (_Ntot * sizeof(float)));
	_out = reinterpret_cast<float*>
		(malloc (_Ntot * sizeof(float)));

	CUDA_SAFE_CALL(	cudaMalloc( (void**) &g_in, 
			sizeof(float)*_Ntot) );

	CUDA_SAFE_CALL( cudaMalloc( (void**) &g_out, 
			sizeof(float)*_Ntot) );

	CUDA_SAFE_CALL( cudaMalloc( (void**) &_IN, 
			sizeof(cufftComplex)*_Ncomplex) );



}

R2C_fft_driver::~R2C_fft_driver(){
	free(_axesDim);
	free(_in);
	free(_out);
	CUDA_SAFE_CALL( cudaFree(g_in));
	CUDA_SAFE_CALL( cudaFree(g_out));
	CUDA_SAFE_CALL( cudaFree(_IN));
}
void R2C_fft_driver::randfill(){
	for(int ii = 0;  ii < _Ntot; ii++)
		_in[ii] = rand()/((float) RAND_MAX);

	CUDA_SAFE_CALL(
	cudaMemcpy(g_in, _in, sizeof(float)*_Ntot, 
			cudaMemcpyHostToDevice)
	);
}
void R2C_fft_driver::initBackend(){
	switch(_Naxes){
		case 1: 
			CUFFT_SAFE_CALL(
				cufftPlan1d( &_r2c, _axesDim[0],
				CUFFT_R2C, 1));
			CUFFT_SAFE_CALL(
				cufftPlan1d( &_c2r, _axesDim[0],
				CUFFT_C2R, 1));
			break;
		case 2:
			CUFFT_SAFE_CALL(
				cufftPlan2d( &_r2c, _axesDim[0],
				_axesDim[1], CUFFT_R2C));
			CUFFT_SAFE_CALL(
				cufftPlan2d( &_c2r, _axesDim[0],
				_axesDim[1], CUFFT_C2R));
			break;
		case 3:
			CUFFT_SAFE_CALL(
				cufftPlan3d( &_r2c, _axesDim[0],
				_axesDim[1], _axesDim[2], CUFFT_R2C));
			CUFFT_SAFE_CALL(
				cufftPlan3d( &_c2r, _axesDim[0],
				_axesDim[1], _axesDim[2], CUFFT_C2R));
			break;
		default:
			std::cerr << "Can handle only 1 to 3 Dim signals!"
				<< std::endl;
			exit(EXIT_FAILURE);
	}
}
void R2C_fft_driver::run(){
	CUFFT_SAFE_CALL(
	cufftExecR2C(_r2c, 
		reinterpret_cast< cufftReal * >(g_in),
		_IN) 
	);
	CUFFT_SAFE_CALL(
	cufftExecC2R(_c2r, _IN,
		reinterpret_cast< cufftReal * >(g_out))
	);
}
void R2C_fft_driver::finalizeBackend(){
	CUFFT_SAFE_CALL( cufftDestroy(_c2r) );
	CUFFT_SAFE_CALL( cufftDestroy(_r2c) );
}


}

