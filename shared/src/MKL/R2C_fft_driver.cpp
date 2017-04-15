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
#include "R2C_fft_driver.h"
using std::complex;
using std::string;

namespace MKL {

  void mklFftCheckStatus(MKL_LONG status, const char *name){

	MKL_LONG predicate = 0;	

	printf("\n------------------------ Check: %s \n\n",name);
	
	predicate = DftiErrorClass ( status , DFTI_MEMORY_ERROR ) ;
	if(predicate != 0){
	printf("DETECT DFTI_MEMORY_ERROR\n");
	}
	
	predicate = DftiErrorClass ( status , DFTI_INVALID_CONFIGURATION ) ;
	if(predicate != 0){
	printf("DETECT DFTI_INVALID_CONFIGURATION\n");
	}
	
	predicate = DftiErrorClass ( status , DFTI_INCONSISTENT_CONFIGURATION ) ;
	if(predicate != 0){
	printf("DETECT DFTI_INCONSISTENT_CONFIGURATION\n");
	}
	
	predicate = DftiErrorClass ( status , DFTI_NUMBER_OF_THREADS_ERROR ) ;
	if(predicate != 0){
	printf("DETECT DFTI_NUMBER_OF_THREADS_ERROR\n");
	}
	
	predicate = DftiErrorClass ( status , DFTI_MULTITHREADED_ERROR ) ;
	if(predicate != 0){
	printf("DETECT DFTI_MULTITHREADED_ERROR\n");
	}
	
	predicate = DftiErrorClass ( status , DFTI_BAD_DESCRIPTOR ) ;
	if(predicate != 0){
	printf("DETECT DFTI_BAD_DESCRIPTOR\n");
	}
	
	predicate = DftiErrorClass ( status , DFTI_UNIMPLEMENTED ) ;
	if(predicate != 0){
	printf("DETECT DFTI_UNIMPLEMENTED\n");
	}
	
	predicate = DftiErrorClass ( status , DFTI_MKL_INTERNAL_ERROR ) ;
	if(predicate != 0){
	printf("DETECT DFTI_MKL_INTERNAL_ERROR\n");
	}
	
	predicate = DftiErrorClass ( status , DFTI_1D_LENGTH_EXCEEDS_INT32 ) ;
	if(predicate != 0){
	printf("DETECT DFTI_1D_LENGTH_EXCEEDS_INT32\n");
	}

	predicate = DftiErrorClass ( status , DFTI_NO_ERROR ) ;
	if(predicate != 0){
	printf("DFTI_NO_ERROR\n");
	}
	
	printf("\n");

}


R2C_fft_driver::R2C_fft_driver(const fftbench_opt& opt) 
: _opt(opt), _in(NULL), _out(NULL), _IN(NULL),
	_Naxes(0), _axesDim(NULL)
{
  int *temp;
	charToBlocks(&temp, &_Naxes, opt.size);			
 
  _axesDim = (MKL_LONG*) malloc(_Naxes * sizeof(MKL_LONG));

  _axesDim[_Naxes-1] = temp[_Naxes];
	_Ncomplex = _axesDim[_Naxes-1] / 2 + 1;
	_Ntot = _axesDim[_Naxes-1];
	
  for(int ii = 0; ii < _Naxes-1; ii++){
    _axesDim[ii] = temp[ii];
		_Ntot *= _axesDim[ii];
		_Ncomplex *= _axesDim[ii];
	}
  free(temp);

	_in = reinterpret_cast< float* > ( 
      mkl_malloc( _Ntot * sizeof(float), MKL::vector_register_size )
      );
  
	_out = reinterpret_cast< float* > ( 
      mkl_malloc( _Ntot * sizeof(float), MKL::vector_register_size )
      );
	
  _IN = reinterpret_cast< complex<float>* > ( 
      mkl_malloc( _Ncomplex * sizeof(float), MKL::vector_register_size )
      );

}

R2C_fft_driver::~R2C_fft_driver(){
	free(_axesDim);
	mkl_free(_in);
	mkl_free(_out);
	mkl_free(_IN);

}
void R2C_fft_driver::randfill(){
	for(int ii = 0;  ii < _Ntot; ii++)
	_in[ii] = rand()/((float) RAND_MAX );

}

void R2C_fft_driver::initBackend(){
  mkl_set_num_threads(_opt.nthreads);
 
  MKL_LONG status = 0;

  if (_Naxes == 1){ //MKL_SINGLE/DOUBE
    status = DftiCreateDescriptor( &_forw, DFTI_SINGLE, DFTI_REAL, 
        _Naxes, _axesDim[0]); 
    mklFftCheckStatus(status," Forward descriptor");
    status = DftiCreateDescriptor( &_back, DFTI_SINGLE, DFTI_REAL, 
        _Naxes, _axesDim[0]); 
    mklFftCheckStatus(status," Reverse descriptor");
  }
  else{
    status = DftiCreateDescriptor( &_forw, DFTI_SINGLE, DFTI_REAL, 
        _Naxes, _axesDim); 
    mklFftCheckStatus(status," Forward descriptor");
    status = DftiCreateDescriptor( &_back, DFTI_SINGLE, DFTI_REAL, 
        _Naxes, _axesDim);  
    mklFftCheckStatus(status," Reverse descriptor");
  }

  status = DftiSetValue( _forw, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  status = DftiSetValue( _forw, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX); //DFTI_REAL_COMPLEX
  
  status = DftiSetValue( _back, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  status = DftiSetValue( _back, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);

  status = DftiCommitDescriptor( _forw );
  status = DftiCommitDescriptor( _back );
}

void R2C_fft_driver::run(){
//  DftiComputeForward( _forw, _in, _IN );
//  DftiComputeBackward( _back, _IN, _out );
#ifdef PRECISIONCHECK
  double num = 0.0, den = 0.0;
  double c = 1.0/_Ntot;
  for (int ii = 0; ii < _Ntot; ii++)
    num += ( _in[ii] - c*_out[ii] ) * ( _in[ii] - c*_out[ii] );

  for (int ii = 0; ii < _Ntot; ii++)
    den +=  _in[ii] * _in[ii];

  printf ("precision %24.16e\n",sqrt(num/den));
  printf ("precision %24.16e\n",den);
  printf ("precision %24.16e\n",num);
#endif
}
void R2C_fft_driver::finalizeBackend(){
  DftiFreeDescriptor( &_forw );
  DftiFreeDescriptor( &_back );
  mkl_free_buffers();
}

}

