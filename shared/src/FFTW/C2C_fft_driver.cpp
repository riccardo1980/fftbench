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
#include "C2C_fft_driver.h"
using std::complex;
using std::string;

namespace FFTW {

C2C_fft_driver::C2C_fft_driver(const fftbench_opt& opt) 
: _opt(opt), _in(NULL), _out(NULL), _IN(NULL),
	_Naxes(0), _axesDim(NULL)
{
	charToBlocks(&_axesDim, &_Naxes, opt.size);			
	_Ntot = _axesDim[0];
	for(int ii = 1; ii < _Naxes; ii++){
		_Ntot *= _axesDim[ii];
	}

	_in = reinterpret_cast< complex<float>* > (fftwf_alloc_complex(_Ntot));
	_out = reinterpret_cast< complex<float>* > (fftwf_alloc_complex(_Ntot));
	_IN = reinterpret_cast< complex<float>* > (fftwf_alloc_complex(_Ntot));

}

C2C_fft_driver::~C2C_fft_driver(){
	free(_axesDim);
	free(_in);
	free(_out);
	free(_IN);
}
void C2C_fft_driver::randfill(){
	for(int ii = 0;  ii < _Ntot; ii++)
	_in[ii] = complex<float>( rand()/((float) RAND_MAX),rand()/((float) RAND_MAX) );

}

void C2C_fft_driver::initBackend(){
	if (_opt.nthreads > 1){
		fftwf_init_threads();
		fftwf_plan_with_nthreads(_opt.nthreads);
	}
	_forw = fftwf_plan_dft(_Naxes, _axesDim,  
			reinterpret_cast<fftwf_complex*>( _in ),
			reinterpret_cast<fftwf_complex*>( _IN ),
		       	FFTW_FORWARD, FFTW_ESTIMATE);
	_back = fftwf_plan_dft(_Naxes, _axesDim,  
			reinterpret_cast<fftwf_complex*>( _IN ),
			reinterpret_cast<fftwf_complex*>( _out ),
		       	FFTW_BACKWARD, FFTW_ESTIMATE);

}

void C2C_fft_driver::run(){
	fftwf_execute_dft(_forw, reinterpret_cast<fftwf_complex*>( _in ), reinterpret_cast<fftwf_complex*>( _IN ));
	fftwf_execute_dft(_back, reinterpret_cast<fftwf_complex*>( _IN ), reinterpret_cast<fftwf_complex*>( _out ));
}
void C2C_fft_driver::finalizeBackend(){
	fftwf_destroy_plan(_forw);
	fftwf_destroy_plan(_back);
	fftwf_cleanup();
}

}

