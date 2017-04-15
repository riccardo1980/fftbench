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

#include "fftbench_mkl_settings.h"

#include "parseUtils.h"
#include "C2C_fft_driver.h"
using std::complex;
using std::string;

namespace MKL {

C2C_fft_driver::C2C_fft_driver(const fftbench_opt& opt) 
: _opt(opt), _in(NULL), _out(NULL), _IN(NULL),
	_Naxes(0), _axesDim(NULL)
{
  int *temp;
	charToBlocks(&temp, &_Naxes, opt.size);			
 

  _axesDim = (MKL_LONG*) malloc(_Naxes * sizeof(MKL_LONG));

  _axesDim[0] = temp[0];
	_Ntot = _axesDim[0];
	for(int ii = 1; ii < _Naxes; ii++){
    _axesDim[ii] = temp[ii];
		_Ntot *= _axesDim[ii];
	}
  free(temp);

	_in = reinterpret_cast< complex<float>* > ( 
      mkl_malloc( _Ntot * sizeof(complex<float>), MKL::vector_register_size )
      );
  
	_out = reinterpret_cast< complex<float>* > ( 
      mkl_malloc( _Ntot * sizeof(complex<float>), MKL::vector_register_size )
      );
	
  _IN = reinterpret_cast< complex<float>* > ( 
      mkl_malloc( _Ntot * sizeof(complex<float>), MKL::vector_register_size )
      );

}

C2C_fft_driver::~C2C_fft_driver(){
	free(_axesDim);
	mkl_free(_in);
	mkl_free(_out);
	mkl_free(_IN);

}
void C2C_fft_driver::randfill(){
	for(int ii = 0;  ii < _Ntot; ii++)
	_in[ii] = complex<float>( rand()/((float) RAND_MAX),rand()/((float) RAND_MAX) );

}

void C2C_fft_driver::initBackend(){
  mkl_set_num_threads(_opt.nthreads);
 
  MKL_LONG status = 0;

  if (_Naxes == 1){
    status = DftiCreateDescriptor( &_forw, DFTI_SINGLE, DFTI_COMPLEX, 
        _Naxes, _axesDim[0]); 
    status = DftiCreateDescriptor( &_back, DFTI_SINGLE, DFTI_COMPLEX, 
        _Naxes, _axesDim[0]); 
  }
  else{
    status = DftiCreateDescriptor( &_forw, DFTI_SINGLE, DFTI_COMPLEX, 
        _Naxes, _axesDim); 
    status = DftiCreateDescriptor( &_back, DFTI_SINGLE, DFTI_COMPLEX, 
        _Naxes, _axesDim);  }
  
    status = DftiSetValue( _forw, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  status = DftiSetValue( _back, DFTI_PLACEMENT, DFTI_NOT_INPLACE);

  status = DftiCommitDescriptor( _forw );
  status = DftiCommitDescriptor( _back );

}

void C2C_fft_driver::run(){
  DftiComputeForward( _forw, _in, _IN );
  DftiComputeBackward( _back, _IN, _out );

}
void C2C_fft_driver::finalizeBackend(){
  DftiFreeDescriptor( &_forw );
  DftiFreeDescriptor( &_back );
  mkl_free_buffers();
}

}

