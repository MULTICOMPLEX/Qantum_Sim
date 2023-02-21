
#ifndef __FFT_HPP__
#define __FFT_HPP__

typedef double real_type;
typedef std::complex<real_type> Complex;
//typedef multicomplex<real_type, 0 > Complex;

#include "vector_operators.hpp"
#include "fftopenmp.h"

inline bool isPowerOfTwo(const size_t num)
{
	return num && (!(num & (num - 1)));
}

void fft(std::vector<Complex>& x, bool inverse = false)
{
	// DFT
	unsigned int N = unsigned(x.size());
	unsigned int k = N, n;
	double thetaT = 3.14159265358979323846264338328L / N;
	if (inverse) thetaT = -thetaT;
	Complex phiT = Complex(cos(thetaT), -sin(thetaT)), T;
	while (k > 1)
	{
		n = k;
		k >>= 1;
		phiT = phiT * phiT;
		T = 1.0L;
		for (unsigned int l = 0; l < k; l++)
		{
			for (unsigned int a = l; a < N; a += n)
			{
				unsigned int b = a + k;
				Complex t = x[a] - x[b];
				x[a] += x[b];
				x[b] = t * T;
			}
			T *= phiT;
		}
	}
	// Decimate
	unsigned int m = (unsigned int)log2(N);

	for (unsigned int a = 0; a < N; a++)
	{
		unsigned int b = a;
		// Reverse bits
		b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
		b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
		b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
		b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
		b = ((b >> 16) | (b << 16)) >> (32 - m);
		if (b > a)
		{
			Complex t = x[a];
			x[a] = x[b];
			x[b] = t;
		}
	}
}

// inverse fft (in-place)
void ifft(std::vector<Complex>& x)
{
	// forward fft
	fft(x, true);

	// scale the numbers
	x /= double(x.size());
}

std::vector<Complex> FFT(std::vector<Complex> x)
{
	fft(x);
	return x;
}

std::vector<Complex> IFFT(std::vector<Complex> x)
{
	ifft(x);
	return x;
}


//2D
void
FFT(std::vector<std::vector<double>>& x, std::vector<std::vector<double>>& y, 
	std::vector<double>& w)
{
	const auto N = int(x.size());

	for (const auto& i : x) {
		if (!isPowerOfTwo(i.size())) {
			std::cout << "not a power of two!";
			exit(0);
		}
	}

	int k;
# pragma omp parallel \
    shared ( x,y,w ) \
    private ( k )
	{
# pragma omp for nowait
		for (k = 0; k < N; k++) {
			cfft2(N, x[k], y[k], w, +1);
			cfft2(N, y[k], x[k], w, +1);
		}
	}

}

//2D
void
IFFT(std::vector<std::vector<double>>& x, std::vector<std::vector<double>>& y,
	std::vector<double>& w)
{	
	const auto N = x.size();
	// forward fft
	FFT(x, y, w);

	// scale the numbers
	for (auto& j : y)
		for (auto& i : j)
			i /= double(N * N);
}


//2D
std::vector<std::vector<Complex>>
FFT(std::vector<std::vector<Complex>> data, bool inverse = false)
{
	const auto N = data.size();

	for (const auto& i : data) {
		if (!isPowerOfTwo(i.size())) {
			std::cout << "not a power of two!";
			exit(0);
		}
	}

	for (auto i = 0; i < N; i++) {
		fft(data[i], inverse);
		fft(data[i], inverse);
	}

	return data;
}

//2D
std::vector<std::vector<Complex>> IFFT(std::vector<std::vector<Complex>> data)
{
	const auto N = data.size();

	// forward fft
	data = FFT(data, true);

	// scale the numbers
	for (auto& j : data)
		for (auto& i : j)
			i /= double(N * N);

	return data;
}


//3D
std::vector<std::vector<std::vector<Complex>>>
FFT(std::vector<std::vector<std::vector<Complex>>> data, bool inverse = false)
{
	const auto N = data.size();

	for (const auto& i : data) {
		for (const auto& k : i) {
			if (!isPowerOfTwo(k.size())) {
				std::cout << "not a power of two!";
				exit(0);
			}
		}
	}

	for (auto k = 0; k < N; k++) {
		for (auto i = 0; i < N; i++) {
			fft(data[k][i], inverse);
			fft(data[k][i], inverse);
			fft(data[k][i], inverse);
		}
	}

	return data;
}

//3D
std::vector<std::vector<std::vector<Complex>>>
IFFT(std::vector<std::vector<std::vector<Complex>>> data)
{
	const auto N = data.size();

	for (const auto& i : data) {
		for (const auto& k : i) {
			if (!isPowerOfTwo(k.size())) {
				std::cout << "not a power of two!";
				exit(0);
			}
		}
	}

	// forward fft
	data = FFT(data, true);

	// scale the numbers
	for (auto& k : data)
		for (auto& j : k)
			for (auto& i : j)
				i /= double(N * N * N);

	return data;
}

template<typename T>
std::vector<T> arange(T start, T stop, T step = 1) {
	std::vector<T> values;
	for (int value = start; value < stop; value += step)
		values.push_back(value);
	return values;
}

template<typename T>
std::vector<double> linspace(T start_in, T end_in, size_t num_in)
{
	std::vector<double> linspaced;

	double start = start_in;
	double end = end_in;
	double num = double(num_in);

	if (num == 0) { return linspaced; }
	if (num == 1)
	{
		linspaced.push_back(start);
		return linspaced;
	}

	auto delta = (end - start) / (num - 1);

	for (auto i = 0; i < num - 1; ++i)
	{
		linspaced.push_back(start + delta * i);
	}
	linspaced.push_back(end); // I want to ensure that start and end
	// are exactly the same as the input
	return linspaced;
}

template<typename dtype>
std::pair< std::vector<std::vector<dtype>>, std::vector<std::vector<dtype>>> meshgrid(
	const std::vector<dtype>& inICoords, const std::vector<dtype>& inJCoords)
{

	auto numRows = inJCoords.size();
	auto numCols = inICoords.size();
	std::vector<dtype> v1(numRows), v2(numRows);

	std::vector < std::vector<dtype>> returnArrayI(numCols, v1);
	std::vector < std::vector<dtype>> returnArrayJ(numCols, v2);

	// first the I array
	for (auto row = 0; row < numRows; ++row)
	{
		for (auto col = 0; col < numCols; ++col)
		{
			returnArrayI[row][col] = inICoords[col];
		}
	}

	// then the I array
	for (auto col = 0; col < numCols; ++col)
	{
		for (auto row = 0; row < numRows; ++row)
		{
			returnArrayJ[row][col] = inJCoords[row];
		}
	}

	return std::make_pair(returnArrayI, returnArrayJ);
}


template <typename T>
	requires	std::same_as<T, std::vector<Complex>>
void cv_to_dv(const T& v1, std::vector<double>& v2)
{
	for (auto j = 0; const auto & i : v1)
	{
		v2[j] = i.real();
		v2[j + 1ull] = i.imag();
		j += 2;
	}
}

template <typename T>
	requires	std::same_as<T, std::vector<Complex>>
void dv_to_cv(const std::vector<double>& v1, T& v2)
{
	for (auto i = 0, j = 0; j < v2.size(); j++)
	{
		v2[j] = Complex(v1[i], v1[i + 1ull]);
		i += 2;
	}
}

void SSFFT(const std::vector<Complex>& Ur, const std::vector<Complex>& Uk, std::vector<Complex>& tmp,
	std::vector<double>& x, std::vector<double>& y, std::vector<double>& w)
{
	const auto N = int(tmp.size());

	int i = 0, j;
# pragma omp for nowait
	for (j = 0; j < N; j++)
	{
		tmp[j] = Ur[j] * tmp[j];
		x[i] = tmp[j].real();
		x[i + 1ull] = tmp[j].imag();
		i += 2;
	}

	cfft2(N, x, y, w, +1);

	i = 0;
# pragma omp for nowait
	for (j = 0; j < N; j++)
	{
		tmp[j] = Complex(y[i], y[i + 1]) * Uk[j];
		x[i] = tmp[j].real();
		x[i + 1ull] = tmp[j].imag();
		i += 2;
	}

	cfft2(N, x, y, w, -1);

	i = 0;
# pragma omp for nowait
	for (j = 0; j < N; j++)
	{
		tmp[j] = Complex(y[i] / N, y[i + 1] / N) * Ur[j];
		i += 2;
	}
}


void SSFFT(const std::vector<Complex>& Ur, const std::vector<Complex>& Uk, std::vector<Complex>& tmp)
{
	tmp = Ur * IFFT(Uk * FFT(Ur * tmp));
}

void Test_FFT()
{
	const auto N = 1024;
	std::vector<Complex> k(N, 0), k1(N, 0), k2(N, 0);
	std::vector<double> w(N, 0), x(N * 2, 0), y(N * 2, 0);
	//std::vector<std::vector<Complex>> j1(N, k);
	std::vector<std::vector<double>> x1(N, x), y1(N, y);
	//std::vector<std::vector<std::vector<Complex>>> psi1(N, j1);

	k[0] = Complex(1.7, 99);

	//psi1 = { {{ 1., 2. },  { 15., 7.} }, { { 1., 9. }, { 15., 7.} } };
	//psi1[0][0][0] = 2;
	//psi1[0][1][0] = 7;
	//psi1[1][1][7] = 37;

	//j1 = { { Complex(1.,2.7), Complex(2.,0) },  { Complex(5.,0), Complex(6.,0) } };

	//j1[1][0] = Complex(777, 2.7);
	//j1[6][6] = Complex(33333, 5);

	cffti(N, w.data());

	cv_to_dv(k, x);
	dv_to_cv(x, k);
	cv_to_dv(k, x);

	fftw_plan planf;
	fftw_plan planr;
	fftw_complex xtw[1024]{}, ytw[1024]{};

	planf = fftw_plan_dft_1d(N, xtw, ytw, FFTW_FORWARD, FFTW_ESTIMATE);
	planr = fftw_plan_dft_1d(N, ytw, xtw, FFTW_BACKWARD, FFTW_ESTIMATE);

	auto begin = std::chrono::high_resolution_clock::now();

	for (auto i = 0; i < 1000; i++) {

		//FFT(k);
		//cfft2(N, x, y, w, +1);
		//cfft2(N, y, x, w, -1);
		fftw_execute(planf);
		fftw_execute(planr);

		for (int j = 0; j < 1024; j++)
		{
			xtw[j][0] /= double(N);
			xtw[j][1] /= double(N);
		}

		//k = IFFT(FFT(k));

	}
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << std::endl << "Took  "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
		<< "[ms]" << std::endl << std::endl;

	//N = 256
	//3D DFT = 3N^2
	//2D DFT = 2N
	//3D/2D  = 384
}

template <typename T>
std::vector<Complex> doDFT(const std::vector<T>& in)
{
	//mxws<uint32_t> rng;
	auto N = in.size();
	Complex	z(0, -2 * std::numbers::pi / N);
	Complex W(exp(z)), Wk(1);

	std::vector<Complex> out(in.size());

	for (auto& x : out)
	{
		Complex	Wkl(1);
		x = 0;

		for (auto& y : in)
		{
			x += y * Wkl;
			//Wkl *= Wk + MX0(rng(-0.0001, 0.0001), rng(-0.0001, 0.0001));
			Wkl *= Wk;
		}
		Wk *= W;
	}
	return out;
}

class DFT_Coeff {
public:
	double Real, Imag;
	DFT_Coeff() {
		Real = 0.0;
		Imag = 0.0;
	}
};

template <typename T>
	requires
std::same_as<T, double>
std::vector<DFT_Coeff> doDFTr(const std::vector<T> function, bool inverse, bool sd,
	std::vector<T>& ds, bool distribution_twiddlefactors) {

	auto N = function.size();

	std::vector<T> sine(N), cosine(N), idft(N);

	std::vector<DFT_Coeff> dft_value(N);

	for (auto n = 0; n < N; n++) {
		cosine[n] = cos(2 * std::numbers::pi * n / N);
		sine[n] = sin(2 * std::numbers::pi * n / N);
	}

	if (distribution_twiddlefactors) {
		cosine = ds;
		std::ranges::rotate(ds, ds.begin() + N / 4ull);
		sine = ds;
	}

	else {
		cosine = sine;
		std::ranges::rotate(sine, sine.begin() + N / 4ull);
	}

	for (auto k = 0; k < N; k++) {
		//std::cout << std::endl;
		dft_value[k].Real = 0;
		dft_value[k].Imag = 0;
		for (auto n = 0; n < N; n++) {
			//cosine[k] = cos(2 * std::numbers::pi * k * n / N);
			//sine[k] = sin(2 * std::numbers::pi * k * n / N);
			auto m = std::modulus()(k * n, N);
			dft_value[k].Real += function[n] * cosine[m];
			dft_value[k].Imag -= function[n] * sine[m];
			//std::cout << k << " " << function[n] * cosine[m] << std::endl;
		}
	}

	if (sd)std::cout << std::endl;
	for (auto j = 0; j < N; j++) {
		//dft_value[j].Real = sqrt(dft_value[j].Real * dft_value[j].Real + dft_value[j].Imag * dft_value[j].Imag);
		//std::cout << dft_value[j].Real << std::endl;
		if (std::abs(dft_value[j].Imag) < 0.00000001) dft_value[j].Imag = 0;
		if (std::abs(dft_value[j].Real) < 0.00000001) dft_value[j].Real = 0;
		//if (sd)std::cout << std::setprecision(8) << dft_value[j].Real << " " << dft_value[j].Imag << std::endl;
	}

	if (inverse) {
		for (auto k = 0; k < N; k++) {
			//std::cout << std::endl;
			idft[k] = 0;
			for (auto n = 0; n < N; n++) {
				auto m = std::modulus()(k * n, N);
				idft[k] += dft_value[n].Real * cosine[m] - dft_value[n].Imag * sine[m];
			}
			idft[k] /= N;
		}

		if (sd) {
			std::cout << std::endl;
			for (auto n = 0; n < N; n++)
				std::cout << idft[n] << " " << function[n] << std::endl;
		}
	}
	return dft_value;
}

template <class T, std::size_t DFT_Length>
class SlidingDFT
{
private:
	/// Are the frequency domain values valid? (i.e. have at elast DFT_Length data
	/// points been seen?)
	bool data_valid = false;

	/// Time domain samples are stored in this circular buffer.
	std::vector<T> x;

	/// Index of the next item in the buffer to be used. Equivalently, the number
	/// of samples that have been seen so far modulo DFT_Length.
	std::size_t x_index = 0;

	/// Twiddle factors for the update algorithm
	std::vector <MX0> twiddle;

	/// Frequency domain values (unwindowed!)
	std::vector<MX0> S;

	bool Hanning_window = false;

public:

	/// Frequency domain values (windowed)
	std::vector<MX0> dft;

	virtual ~SlidingDFT() = default;

	T damping_factor = std::nexttoward((T)1, (T)0);

	/// Constructor
	SlidingDFT()
	{
		x.resize(DFT_Length);
		twiddle.resize(DFT_Length);
		S.resize(DFT_Length);
		dft.resize(DFT_Length);

		const MX0 j(0, 1);
		auto N = DFT_Length;

		// Compute the twiddle factors, and zero the x 
		for (auto k = 0; k < DFT_Length; k++) {
			T factor = 2 * std::numbers::pi * k / N;
			twiddle[k] = exp(j * factor);
		}
	}

	/// Determine whether the output data is valid
	bool is_data_valid()
	{
		return data_valid;
	}

	/// Update the calculation with a new sample
	/// Returns true if the data are valid (because enough samples have been
	/// presented), or false if the data are invalid.
	bool update(T new_x)
	{
		// Update the storage of the time domain values
		const T old_x = x[x_index];
		x[x_index] = new_x;

		// Update the DFT
		const T r = damping_factor;
		const T r_to_N = pow(r, (T)DFT_Length);
		for (auto k = 0; k < DFT_Length; k++)
			S[k] = twiddle[k] * (r * S[k] - r_to_N * old_x + new_x);

		if (Hanning_window) {
			// Apply the Hanning window
			dft[0] = (T)0.5 * S[0] - (T)0.25 * (S[DFT_Length - 1] + S[1]);
			for (size_t k = 1; k < (DFT_Length - 1); k++) {
				dft[k] = (T)0.5 * S[k] - (T)0.25 * (S[k - 1] + S[k + 1]);
			}
			dft[DFT_Length - 1] = (T)0.5 * S[DFT_Length - 1] - (T)0.25 * (S[DFT_Length - 2] + S[0]);
		}
		else
			dft = S;

		// Increment the counter
		x_index++;
		if (x_index >= DFT_Length) {
			data_valid = true;
			x_index = 0;
		}

		// Done.
		return data_valid;
	}
};

template <typename T>
void slidingDFT_driver(
	std::vector<T>& Y,
	std::vector<MX0>& cx
)
{
	const auto N = 2048;
	SlidingDFT<T, N> dft;

	for (size_t i = 0; i < N; i++) {
		dft.update(Y[i]);
		if (dft.is_data_valid()) {
			for (size_t j = 0; j < N; j++)
				cx[j] = dft.dft[j];
		}
	}
}

template<typename T>
std::vector<T> fftshift(const std::vector<T>& k) {
	std::vector<T> v = k;
	std::ranges::rotate(v, v.begin() + int(round(v.size() / 2.)));
	return v;
}

template<typename T>
std::vector<T> ifftshift(const std::vector<T>& k) {
	std::vector<T> v = k;
	std::ranges::rotate(v, v.end() - int(round(v.size() / 2.)));
	return v;
}



std::vector<double> fftfreq(int n, double d = 1.0) {
	auto val = 1.0 / (n * d);
	std::vector<double> results(n);
	auto N = (n - 1) / 2 + 1;
	auto p1 = arange<int>(0, N);
	for (auto x = 0; x < N; x++)
		results[x] = p1[x];
	auto p2 = arange<int>(-(n / 2), 0);
	for (auto x = N; x < results.size(); x++)
		results[x] = p2[x - size_t(N)];
	for (auto x = 0; x < results.size(); x++)
		results[x] *= val;
	return results;
}

std::vector<double> init_fftfreq(int N, double dx, double hbar) {
	std::vector<double> px = fftfreq(N, dx);
	for(auto & i : px)
		i *= hbar * 2 * std::numbers::pi;
	return px * px;
}


std::vector<float> fftfreq(int n, float d = 1.0) {
	float val = 1.0f / (n * d);
	std::vector<float> results(n);
	int N = (n - 1) / 2 + 1;
	auto p1 = arange<int>(0, N);
	for (auto x = 0; x < N; x++)
		results[x] = float(p1[x]);
	auto p2 = arange<int>(-(n / 2), 0);
	for (auto x = N; x < results.size(); x++)
		results[x] = float(p2[x - size_t(N)]);
	for (auto x = 0; x < results.size(); x++)
		results[x] *= val;
	return results;
}


std::vector<float> init_fftfreq(int N, float dx, float hbar) {
	std::vector px = fftfreq(N, dx);
	for (auto& i : px)
		i *= hbar * 2.0f * float(std::numbers::pi);
	return px * px;
}

#endif // __FFT_HPP__ 