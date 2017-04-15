#include <sstream>
using std::string;


void charToBlocks(int **axesDim, int *Naxes, string splitting){
string token;
const char delim = 'x';

std::istringstream iss(splitting);

	*Naxes = std::count(splitting.begin(),splitting.end(),'x') + 1;

	*axesDim = (int*) malloc(*Naxes * sizeof(int));

	int ii = 0;
	while ( getline(iss, token, delim) ){
		(*axesDim)[ii] = atoi(token.c_str());
		ii++;
	}
}	

