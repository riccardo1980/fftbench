function [tn] = fft_driver(S, varargin)
% tn = fft_driver(S)
% measures fft+ifft time, given input size S
% time is averaged ( see LOOPS optional parameter )
% warmup loop is performed before measure loops.
%
% optional parameters (key,value pairs):
% LOOPS 
%	number of loops to run
%	default = 100
% SEED
%	seed for rng
%	default = 123
% TYPE 
%	type of the transforms:
%	SH 	single precision real input, hermitianity of
%		it's fourier transform is exploited 
%	SC 	single precision real input, backward transform
%		is complex to complex, only the real part is retained
%
%	DH 	double precision real input, hermitianity of
%		it's fourier transform is exploited 
%	DZ 	double precision real input, backward transform
%		is complex to complex, only the real part is retained
%
%	D	double precision complex input, double precision complex output
%	Z	single precision complex input, single precision complex output	
%
%	default = 'DH' 
% VERBOSE
%	verbosity level:
%	0 	print nothing
%	1 	print string type
%	2	 adds storage statistics
%	default = 0

% Authors: 
%   Riccardo Zanella
%    Dept. of Mathematics, University of Ferrara, Italy
%    riccardo.zanella@unife.it
%
% Copyright (C) 2011 by R. Zanella
% ------------------------------------------------------------------------------
% COPYRIGHT NOTIFICATION
%
% Permission to copy and modify this software and its documentation for 
% internal research use is granted, provided that this notice is retained 
% thereon and on all copies or modifications. The authors and their
% respective Universities makes no representations as to the suitability 
% and operability of this software for any purpose. It is provided "as is"
% without express or implied warranty. Use of this software for commercial
% purposes is expressly prohibited without contacting the authors.
%
% This program is free software; you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by the
% Free Software Foundation; either version 3 of the License, or (at your 
% option) any later version.
%
% This program is distributed in the hope that it will be useful, but 
% WITHOUT ANY WARRANTY; without even the implied warranty of 
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
% See the GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License along 
% with this program; if not, either visite http://www.gnu.org/licenses/
% or write to
% Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
% ==============================================================================


% default parameters
LOOPS   = 100;
SEED    = 123;
TYPE    = 'DH';
VERBOSE = 0;


validTypes   = { 'SH','SC','DH','DZ','C','Z' };
descriptions = { 'single precision real input, hermitian', ...
		 'single precision real input, cast', ...
		 'double precision real input, hermitian', ...
		 'double precision real input, cast', ...
		 'single precision complex input', ...
		 'double precision complex input'};

if (rem(length(varargin),2)==1)
	error('Optional parameters should always go by pair');
else
	for(ii=1:2:(length(varargin)-1))
		switch( upper(varargin{ii}) )
			case 'LOOPS'
				LOOPS = varargin{ii+1};
			case 'SEED'
				SEED = varargin{ii+1};
			case 'TYPE'
				TYPE = varargin{ii+1};
			case 'VERBOSE'
				VERBOSE = varargin{ii+1};
			otherwise
				error(['unrecognized option: ''' varargin{ii} '''']);
		end
	end
end

%%%%%%%%%%%%%%%%%%%%
% parameter checking
%%%%%%%%%%%%%%%%%%%%
if ( LOOPS <0 )
	error('LOOPS number must be positive');
end

if ( numel(S) == 1)
	S = [S 1]; 
end
% seed ?
numericType = find( strcmp(TYPE, validTypes) == 1 );
if ( isempty (numericType) )
	error(['TYPE must be one of the following: ' sprintf('%s ',validTypes{:}) ]);
end


%%%%%%%%%%%%%%%%%%%%
% parameter checking
%%%%%%%%%%%%%%%%%%%%
if (VERBOSE >0)
	disp(descriptions{numericType})
end

%% set function handles

%% generation
switch (TYPE)
	case { 'SH','SC' }
		generate  = @(x) ( single(rand(x)) );
	case { 'DH','DZ' }
		generate  = @(x) ( rand(x) );
	case { 'C','C' }
		generate  = @(x) ( complex(single(rand(x)),single(rand(x))) );
	case { 'Z','Z' }
		generate  = @(x) ( complex(rand(x),rand(x)) );

end

forward = @(x) ( fftn(x) ); 

%% backward 
switch (TYPE)
	case {'SH','DH'}
		backward  = @(x) ( ifftn(x,'symmetric') );
	case {'SC','DZ'}
		backward  = @(x) ( real(ifftn(x)) );
	case {'C','Z'}
		backward  = @(x) ( ifftn(x) );
end

%%%%%%%%%%%%%%%%%%%%
% data generation
%%%%%%%%%%%%%%%%%%%%

rng(SEED);
img = generate(S);
if ( VERBOSE > 1 )
	disp(['After Generation'])
	whos('img')
end

%%%%%%%%%%%%%%%%%%%%
% warm up stage
%%%%%%%%%%%%%%%%%%%%
TF  = forward(img);
if ( VERBOSE > 1 )
	disp(['Forward'])
	whos('TF')
end

img = backward(TF);
if ( VERBOSE > 1 )
	disp(['Reverse'])
	whos('img')
end

%%%%%%%%%%%%%%%%%%%%
% Test loop
%%%%%%%%%%%%%%%%%%%%
tic;
for ii = 1 : LOOPS,
	TF  = forward(img);
	img = backward(TF);
end
tn = toc / LOOPS;

