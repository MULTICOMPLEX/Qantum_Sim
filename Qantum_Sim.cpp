#include "functions.hpp"

int main(int argc, char** argv)
{
	std::setlocale(LC_ALL, "en_US.utf8");
	
	Quantum* qm{};

	qm = new(Quantum);

	//Test_FFT();
  //qm->harmonic_oscillator_plus_coulomb_interaction<double>();
  //std::cout << qm->initial_wavefunction2D()[0];

  delete(qm);

  //Test_Maxwell();


	return (0);
}

