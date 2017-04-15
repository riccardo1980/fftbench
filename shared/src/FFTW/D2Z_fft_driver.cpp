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
#include <cstdlib>

#include "parseUtils.h"
#include "D2Z_fft_driver.h"
using std::complex;
using std::string;

namespace FFTW {

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

	_in = fftw_alloc_real(_Ntot);
	_out = fftw_alloc_real(_Ntot);
	_IN = reinterpret_cast< complex<double>* > (fftw_alloc_complex(_Ncomplex));
};
D2Z_fft_driver::~D2Z_fft_driver(){
	free(_axesDim);
	free(_in);
	free(_out);
	free(_IN);
};
void D2Z_fft_driver::randfill(){
	for(int ii = 0;  ii < _Ntot; ii++)
		_in[ii] = rand()/((double) RAND_MAX);
};
void D2Z_fft_driver::initBackend(){
	if (_opt.nthreads > 1){
		fftw_init_threads();
		fftw_plan_with_nthreads(_opt.nthreads);
	}
	_r2c = fftw_plan_dft_r2c(_Naxes, _axesDim, _in, 
			reinterpret_cast<fftw_complex*>( _IN ), FFTW_ESTIMATE);
	_c2r = fftw_plan_dft_c2r(_Naxes, _axesDim, 
			reinterpret_cast<fftw_complex*>( _IN ), _out, FFTW_ESTIMATE);
};
void D2Z_fft_driver::run(){
	fftw_execute_dft_r2c(_r2c, _in, reinterpret_cast<fftw_complex*>( _IN ));
	fftw_execute_dft_c2r(_c2r, reinterpret_cast<fftw_complex*>( _IN ), _out);
};
void D2Z_fft_driver::finalizeBackend(){
	fftw_destroy_plan(_c2r);
	fftw_destroy_plan(_r2c);
	fftw_cleanup();
};

}

