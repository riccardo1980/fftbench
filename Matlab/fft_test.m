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

close all
clear all
% test time for fft
TYPES     = { 'SH', 'SC', 'DH','DZ', 'C', 'Z' };
TESTCASES = { 'A', 'B', 'C', 'D' };
LOOPS = 100;

for TT=1:numel(TYPES),
	
	TYPE = TYPES{TT};
	fprintf('\n%s\n', TYPE);

	for CC=1:numel(TESTCASES),

		TESTCASE = TESTCASES{CC};
		switch (TESTCASE)
			case 'A'
				testSizes = { [   1048576], [  1024   1024], [  256   256   16] };
			case 'B'
				testSizes = { [ 4*1048576], [2*1024 2*1024], [  256   256 4*16] };
			case 'C'
				testSizes = { [16*1048576], [4*1024 4*1024], [2*256 2*256 4*16] };
			case 'D'
				testSizes = { [64*1048576], [8*1024 8*1024], [4*256 4*256 4*16] };
		end

		N = numel(testSizes);
		testRes = cell(N,1);

		for ii=1:N,
			thisSize = testSizes{ii};
			s = sprintf('%d', thisSize(1));
			for jj=2:numel(thisSize),
				s = sprintf('%sx%d', s, thisSize(jj));
			end
			testRes{ii}.s = s;
			testRes{ii}.t = fft_driver(thisSize, 'LOOPS', LOOPS, 'TYPE', TYPE);
			
			fprintf('%s \t %d\n', testRes{ii}.s, testRes{ii}.t  );
		end
	
	end
end
