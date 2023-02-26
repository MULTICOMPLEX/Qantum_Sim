
#ifndef __FUNCTIONS_HPP__
#define __FUNCTIONS_HPP__

//https://www.algorithm-archive.org/contents/split-operator_method/split-operator_method.html

#include <future>
#include <numbers>
#include "MULTICOMPLEX.hpp"
#include "matplotlib.hpp"
#include "constants.hpp"
#include "fftw3.h"
#include "fft.hpp"
#include "nlohmann/json.hpp"

class Quantum
{

private:
	plot_matplotlib plot;
	using json = nlohmann::json;
	using clock = std::chrono::high_resolution_clock;
	std::pair< std::vector<std::vector<double>>, std::vector<std::vector<double>>> x3;

public:

	static constexpr int N = 2048;

	///modes :
	//"tunnel+", simulate tunneling
	//"tunnel-", simulate tunneling
	//"two tunnel+", simulate tunneling
	//"two tunnel+-", simulate tunneling
	//"diode", simulate diode
	//"capacitor", simulate capacitor
	//"free", no barrier

	json js = {
	{"name", "Q0"},
	{"mode", "two tunnel+-"},
	{"total time", 1.5 * qc::femtoseconds},
	{"store steps", 200},
	{"sigma", 0.7 * qc::Am}, //0.2
	{"v0", 40}, //initial_wavefunction amplitude
	{"V0", 2}, //barrier voltage 
	{"initial offset", -15},
	{"N", N},
	{"dt", 2 / (log2(N) * sqrt(N))},
	{"x0", 0}, //	//barrier x
	{"x1", 3},
	{"x2", 12},
	{"extent", 25 * qc::Am},
	{"extentN", -75 * qc::Am},
	{"extentP", +85 * qc::Am},
	{"imaginary time evolution", false},
	{"animation duration", 4}, //seconds
	{"save animation", true},
	{"fps", 30},
	{"path save", "../gifs/"}
	};

	const double dt_store = js["total time"].get<double>() / js["store steps"].get<int>();
	const int Nt_per_store_step = int(round(dt_store / js["dt"].get<double>()));

	fftwf_plan planf;
	fftwf_plan planr;

	fftwf_complex xtw[N]{}, ytw[N]{};

	virtual ~Quantum() {
		fftwf_destroy_plan(planf);
		fftwf_destroy_plan(planr);
	}

	std::vector<std::vector<Complex>> psi;

	Quantum()
	{

		sav_settings(js, js["name"]);
		get_settings(js, js["name"]);

		//mxws<uint32_t> rng;
		//std::vector<double> w(N, 0), x(N * 2, 0), y(N * 2, 0);
		//cffti(N, w.data());

		planf = fftwf_plan_dft_1d(N, xtw, ytw, FFTW_FORWARD, FFTW_ESTIMATE);
		planr = fftwf_plan_dft_1d(N, xtw, ytw, FFTW_BACKWARD, FFTW_ESTIMATE);

		psi.resize(js["store steps"]);

		const std::vector<double> X = linspace(js["extentN"], js["extentP"], N);
		const std::vector<double> Vgrid = potential_barrier(X, js["x0"], js["x1"], js["x2"]);

		const auto dx = X[1] - X[0];
		const std::vector<double> FFTfreq = init_fftfreq(js["N"], dx, qc::hbar);

		psi[0] = initial_wavefunction(X, js["sigma"], js["v0"], js["initial offset"]);

		std::vector<Complex> Ur(Vgrid.size()), Uk(FFTfreq.size());
		auto m = 1.;
		init_Uk_Ur(Vgrid, FFTfreq, Ur, Uk, js["dt"], js["imaginary time evolution"], m);

		//std::cout << "Error O(dt^3) = " << pow(dt, 3) << std::endl;
		std::cout << "store steps " << js["store steps"] << std::endl;
		std::cout << "Nt_per_store_step " << Nt_per_store_step << std::endl;

		auto begin = clock::now();

		for (auto i = 1; i < js["store steps"]; i++) {

			auto tmp = psi[i - 1ull];
			for (auto j = 0; j < Nt_per_store_step; j++) {

				//SSFFT(Ur, Uk, tmp);
				//SSFFT(Ur, Uk, tmp, x, y, w);
				SSFFTW<float>(Ur, Uk, tmp);

				if (js["imaginary time evolution"])
					tmp /= amax(tmp);
			}
			psi[i] = tmp;
		}

		auto end = clock::now();
		std::cout << std::endl << "Took  "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
			<< "[ms]" << std::endl << std::endl;

		psi /= amax(psi);

		plotPSI(X, js["save animation"]);

	}

	template <typename T>
	void init_Uk_Ur(const std::vector<T>& Vgrid, const std::vector<T>& FFTfreq,
		std::vector<Complex>& Ur, std::vector<Complex>& Uk, double dt, bool ite, double m)
	{
		Complex J(0, 1);
		if (ite)
			J = 1;

		Ur = exp(-0.5 * J * (dt / qc::hbar) * Vgrid);
		Uk = exp(-0.5 * J * (dt / (m * qc::hbar)) * FFTfreq);
	}

	template <typename T>
	void plotPSI(const std::vector<T>& x, bool save) {

		plot.Py_STR("ndim = 1");
		plot.Py_STR("N = " + std::to_string(N) + "");
		plot.Py_STR("store_steps = " + std::to_string(psi.size()) + "");

		plot.Py_STR("psi = np.zeros((store_steps, *([N] * ndim)), dtype = np.complex128)");

		std::vector<double> kr(psi[0].size()), ki(psi[0].size());
		int j = 0;
		for (const auto& k : psi) {
			for (auto i = 0; i < k.size(); i++)
			{
				kr[i] = k[i].real();
				ki[i] = k[i].imag();
			}
			plot.Py_STR("psi[" + std::to_string(j) + "].real = np.array([" + plot.vector_data(kr) + "])");
			plot.Py_STR("psi[" + std::to_string(j) + "].imag = np.array([" + plot.vector_data(ki) + "])");
			j++;
		}

		plot.Py_STR("px = 1 / plt.rcParams['figure.dpi']");
		plot.Py_STR("figsize = (900*px, 556*px)");
		plot.Py_STR("fig = plt.figure(figsize = figsize, facecolor='#002b36')");

		auto a = std::to_string(js["x0"] - 0.5);
		a += ",";
		a += std::to_string(js["x0"] + 0.5);
		auto b = std::to_string(js["x1"] - 0.5);
		b += ",";
		b += std::to_string(js["x1"] + 0.5);
		auto c = std::to_string(js["x2"] - 0.5);
		c += ",";
		c += std::to_string(js["x2"] + 0.5);

		plot.Py_STR("ax = plt.gca()");

		auto aa = std::to_string(js["x0"] - 2.5);

		if (js["mode"] == "tunnel+") {
			plot.Py_STR("plt.axvspan(" + a + ", alpha = 0.5, color = 'red')");
			plot.Py_STR("plt.text(" + aa + ", 0.95, '+', horizontalalignment='center',\
				verticalalignment = 'center', color = 'red')");
		}

		if (js["mode"] == "tunnel-") {
			plot.Py_STR("plt.axvspan(" + a + ", alpha = 0.5, color = 'gray')");
			plot.Py_STR("plt.text(" + aa + ", 0.95, '-', horizontalalignment='center',\
				verticalalignment = 'center', color = 'gray')");
		}

		auto ab = std::to_string(js["x1"] + 2.5);
		auto ac = std::to_string(js["x2"] + 2.5);


		if (js["mode"] == "two tunnel+") {
			plot.Py_STR("plt.axvspan(" + a + ", alpha = 0.5, color = 'red')");
			plot.Py_STR("plt.text(" + aa + ", 0.95, '+', horizontalalignment='center',\
				verticalalignment = 'center', color = 'red')");

			plot.Py_STR("plt.axvspan(" + b + ", alpha = 0.5, color = 'red')");
			plot.Py_STR("plt.text(" + ab + ", 0.95, '+', horizontalalignment='center',\
				verticalalignment = 'center', color = 'red')");
		}

		if (js["mode"] == "two tunnel+-") {
			plot.Py_STR("plt.axvspan(" + a + ", alpha = 0.5, color = 'red')");
			plot.Py_STR("plt.text(" + aa + ", 0.95, '+', horizontalalignment='center',\
				verticalalignment = 'center', color = 'red')");

			plot.Py_STR("plt.axvspan(" + b + ", alpha = 0.5, color = 'gray')");
			plot.Py_STR("plt.text(" + ab + ", 0.95, '-', horizontalalignment='center',\
				verticalalignment = 'center', color = 'gray')");
		}

		if (js["mode"] == "diode") {
			plot.Py_STR("plt.axvspan(" + a + ", alpha = 0.5, color = 'red')");

			plot.Py_STR("plt.text(" + aa + ", 0.95, '+', horizontalalignment='center',\
				verticalalignment = 'center', color = 'red')");

			plot.Py_STR("plt.axvspan(" + b + ", alpha = 0.5, color = 'gray')");

			plot.Py_STR("plt.text(" + ab + ", 0.95, '-', horizontalalignment='center',\
				verticalalignment = 'center', color = 'gray')");

			plot.Py_STR("plt.axvspan(" + c + ", alpha = 0.5, color = 'gray')");

			plot.Py_STR("plt.text(" + ac + ", 0.95, '-', horizontalalignment='center',\
				verticalalignment = 'center', color = 'gray')");

		}

		if (js["mode"] == "capacitor") { //or tunnel two barrier
			plot.Py_STR("plt.axvspan(" + a + ", alpha = 0.5, color = 'gray')");
			plot.Py_STR("plt.text(" + aa + ", 0.95, '-', horizontalalignment='center',\
				verticalalignment = 'center', color = 'gray')");

			//plot.Py_STR("plt.axvspan(" + b + ", alpha = 0.2, color = 'red')");
			plot.Py_STR("plt.axvspan(" + c + ", alpha = 0.5, color = 'gray')");
			plot.Py_STR("plt.text(" + ac + ", 0.95, '-', horizontalalignment='center',\
				verticalalignment = 'center', color = 'gray')");
		}

		plot.Py_STR("plt.ylim(-1, 1)");

		std::vector<double> X = x;
		X /= qc::Am;

		plot.Py_STR("x = np.array([" + plot.vector_data(X) + "])");


		plot.Py_STR("" + utf8_encode(u8"Å = 1.8897261246257702") + "");

		plot.Py_STR("ax.set_xlabel('$" + utf8_encode(u8"[Å]") + "$')");

		std::string title = "$\\psi(x,t)$ ";

		if (js["mode"] == "tunnel+")
			title += "tunnel+";

		if (js["mode"] == "tunnel-")
			title += "tunnel-";

		if (js["mode"] == "two tunnel+")
			title += "two tunnel+";

		if (js["mode"] == "two tunnel+-")
			title += "two tunnel+-";

		if (js["mode"] == "diode")
			title += "diode";

		if (js["mode"] == "capacitor")
			title += "capacitor";

		if (js["mode"] == "free")
			title += "";

		plot.Py_STR("ax.set_title(\"" + title + "\", color = 'white')");

		plot.Py_STR("ax.xaxis.label.set_color('white')\n\
ax.yaxis.label.set_color('white')\n\
ax.tick_params(colors = 'white')\n\
ax.spines['left'].set_color('white')\n\
ax.spines['bottom'].set_color('white')\n\
ax.spines['top'].set_color('white')\n\
ax.spines['right'].set_color('white')");

		plot.Py_STR("index = 0");
		plot.Py_STR("real_plot, = ax.plot(x, np.real(psi[index]), label = '$Re|\\psi(x)|$')");
		plot.Py_STR("imag_plot, = ax.plot(x, np.imag(psi[index]), label = '$Im|\\psi(x)|$')");
		plot.Py_STR("abs_plot, = ax.plot(x, np.abs(psi[index]), label = '$|\\psi(x)|$')");
		plot.Py_STR("ax.legend(loc = 'lower left')");

		plot.Py_STR("ax.set_facecolor('#002b36')\n\
leg = ax.legend(facecolor = '#002b36', loc = 'lower left')\n\
for line, text in zip(leg.get_lines(), leg.get_texts()) :\
				text.set_color(line.get_color())");

		plot.Py_STR("femtoseconds = 4.134137333518212 * 10.");

		plot.Py_STR("animation_duration = " + std::to_string(js["animation duration"].get<double>()) + "");

		plot.Py_STR("fps = " + std::to_string(js["fps"].get<int>()) + "");

		plot.Py_STR("total_frames = int(fps * animation_duration)");

		plot.Py_STR("total_time = " + std::to_string(js["total time"].get<double>()) + "");
		plot.Py_STR("dt = total_time / total_frames");

		plot.Py_STR("time_ax = ax.text(0.97, 0.97, \"\", color = \"white\",\
			transform = ax.transAxes, ha = \"right\", va = \"top\")");
		plot.Py_STR("xdt = np.linspace(0, " + std::to_string(js["total time"].get<double>()) + " / femtoseconds, total_frames)");
		plot.Py_STR("psi_index = np.linspace(0, " + std::to_string(js["store steps"].get<int>()-1) + ", total_frames)");
		plot.Py_STR("def func_animation(frame) :\n\
		index = int(psi_index[frame])\n\
		time_ax.set_text(u\"t = {} femtoseconds\".format(\" % .3f\" % (xdt[frame])))\n\
		real_plot.set_ydata(np.real(psi[index]))\n\
		imag_plot.set_ydata(np.imag(psi[index]))\n\
		abs_plot.set_ydata(np.abs(psi[index]))\n\
		return");
		
		
		plot.Py_STR("from matplotlib import animation");
		plot.Py_STR("a = animation.FuncAnimation(fig, func_animation, \
			blit = False, frames = total_frames, interval= 1/fps * 1000)");

		std::string jss = js["path save"];
		jss += js["mode"];
		if (save) {
			plot.Py_STR("a.save('" + jss + ".gif', fps = fps, metadata = dict(artist = 'Me'))");
		}

		plot.show();
	}

	void to_json(json& j, const std::string p, const double x) {
		j = json{ {p, x} };
	}

	bool check_if_file_exists(std::string name) {
		std::ifstream ifile;
		bool b = false;
		ifile.open(name + ".json");
		if (ifile) b = true;
		ifile.close();
		return b;
	}

	void sav_settings(json& js, std::string name) {
		std::ofstream outf(name + ".json");
		if (!check_if_file_exists(name + ".json"))
		{
			outf << js;
			outf.close();
		}
	}

	void get_settings(json& js, std::string name) {
		std::ifstream inf(name + ".json");
		if (!inf) {
			std::cout << "file does not exist !";
			exit(0);
		}
		inf >> js;
		inf.close();
	}

	template <typename T>
	void SSFFTW(const std::vector<Complex>& Ur, const std::vector<Complex>& Uk, std::vector<Complex>& tmp)
	{
		const auto N = int(tmp.size());

		int j;

		for (j = 0; j < N; j++)
		{
			tmp[j] = Ur[j] * tmp[j];
			xtw[j][0] = T(tmp[j].real());
			xtw[j][1] = T(tmp[j].imag());
		}

		fftwf_execute(planf);

		for (j = 0; j < N; j++)
		{
			tmp[j] = Complex(ytw[j][0], ytw[j][1]) * Uk[j];
			xtw[j][0] = T(tmp[j].real());
			xtw[j][1] = T(tmp[j].imag());
		}

		fftwf_execute(planr);

		for (j = 0; j < N; j++)
		{
			tmp[j] = Complex(ytw[j][0] / N, ytw[j][1] / N) * Ur[j];
		}
	}


	template <typename T>
		requires std::same_as<T, std::vector<std::vector<Complex>>>
	double amax(const T& a) {
		std::vector<double> tmp;

		for (const auto& y : a)
			for (const auto& i : y)
				tmp.push_back(abs(i));

		auto val = std::ranges::max_element(tmp.begin(), tmp.end());
		return *val;
	}

	template <typename T>
	double amax(const T& a) {
		std::vector<double> tmp;

		for (const auto& i : a)
			tmp.push_back(abs(i));

		auto val = std::ranges::max_element(tmp.begin(), tmp.end());
		return *val;
	}

	template <typename T>
	double amin(const T& a) {
		std::vector<double> tmp;

		for (const auto& y : a)
			for (const auto& i : y)
				tmp.push_back(abs(i));

		auto val = std::ranges::min_element(tmp.begin(), tmp.end());
		return *val;
	}

	template <typename T>
	std::vector<Complex> initial_wavefunction(const std::vector<T>& x, double sigma = 0.7 * qc::Am, double v0 = 40,
		double offset = 20) {
		//This wavefunction correspond to a gaussian wavepacket with a mean X momentum equal to p_x0
		v0 *= qc::Am / qc::femtoseconds;
		auto p_x0 = qc::m_e * v0;
		offset = -offset;
		std::vector<Complex> v;
		Complex j(0, 1);

		v = exp(-1 / (4 * pow(sigma, 2)) * (pow(x + offset, 2)) / sqrt(2 * std::numbers::pi * pow(sigma, 2))) *
			exp(p_x0 * x * j);

		return v;
	}

	template <typename T>
	std::vector < std::vector<T>> harmonic_oscillator_plus_coulomb_interaction()
	{
		//double extent = 25 * qc::Am;
		auto x1 = linspace(-js["extent"].get<double>() / 2, js["extent"].get<double>() / 2, js["N"]);
		auto x2 = linspace(-js["extent"].get<double>() / 2, js["extent"].get<double>() / 2, js["N"]);
		x3 = meshgrid(x1, x2);
		double k = 0.5;
		std::vector < std::vector<T>> V_harmonic = 0.5 * k * pow(x3.first, 2) + 0.5 * k * pow(x3.second, 2);
		k = 30.83;
		std::vector < std::vector<T>> r = abs(x3.first - x3.second);
		r = where(r < 0.0001, 0.0001, r);
		std::vector < std::vector<T>> V_coulomb_interaction = k / r;
		return V_harmonic + V_coulomb_interaction;
	}

	std::vector < std::vector<double>> initial_wavefunction2D(double sigma = 0.7, double mu01 = 1, double mu02 = 1
	) {
		//This wavefunction correspond to two stationary gaussian wavepackets.The wavefunction must be symmetric : Ψ(x1, x2) = Ψ(x2, x1)
		sigma = 0.4 * qc::Am;
		auto x1 = x3.first;
		auto x2 = x3.second;
		mu01 = -7.0 * qc::Am;
		mu02 = 0.0 * qc::Am;

		return (exp(-pow(x1 - mu01, 2) / (4 * pow(sigma, 2))) * exp(-pow(x2 - mu02, 2) / (4 * pow(sigma, 2)))
			+ exp(-pow(x1 - mu02, 2) / (4 * pow(sigma, 2))) * exp(-pow(x2 - mu01, 2) / (4 * pow(sigma, 2))));
	}

	template <typename T>
	std::vector<T> potential_barrier(const std::vector<T>& x, double x0, double x1, double x2) {
		using namespace std;
		using std::ranges::views::zip;

		auto a = 1 * qc::Am;

		std::vector<T> barrier(x.size());

		if (js["mode"] == "tunnel+") {
			for (const auto& i : zip(x, barrier)) {
				if (((get<0>(i) > (js["x0"] * qc::Am - a / 2)) && (get<0>(i) < (js["x0"] * qc::Am + a / 2))))
					get<1>(i) = js["V0"];
			}
		}

		if (js["mode"] == "tunnel-") {
			for (const auto& i : zip(x, barrier)) {
				if (((get<0>(i) > (js["x0"] * qc::Am - a / 2)) && (get<0>(i) < (js["x0"] * qc::Am + a / 2))))
					get<1>(i) = -js["V0"];
			}
		}

		if (js["mode"] == "two tunnel+") {
			for (const auto& i : zip(x, barrier)) {
				if (((get<0>(i) > (js["x0"] * qc::Am - a / 2)) && (get<0>(i) < (js["x0"] * qc::Am + a / 2))))
					get<1>(i) = js["V0"];
				if (((get<0>(i) > (js["x1"] * qc::Am - a / 2)) && (get<0>(i) < (js["x1"] * qc::Am + a / 2))))
					get<1>(i) = js["V0"];
			}
		}


		if (js["mode"] == "two tunnel+-") {
			barrier = where((x > (js["x0"] * qc::Am - a / 2)) & (x < (js["x0"] * qc::Am + a / 2)), js["V0"], barrier);
			barrier = where((x > (js["x1"] * qc::Am - a / 2)) & (x < (js["x1"] * qc::Am + a / 2)), -js["V0"], barrier);
		}

		if (js["mode"] == "diode") {
			for (const auto& i : zip(x, barrier)) {
				if (((get<0>(i) > (js["x0"] * qc::Am - a / 2)) && (get<0>(i) < (js["x0"] * qc::Am + a / 2))))
					get<1>(i) = js["V0"];
				if (((get<0>(i) > (js["x1"] * qc::Am - a / 2)) && (get<0>(i) < (js["x1"] * qc::Am + a / 2))))
					get<1>(i) = -js["V0"];
				if (((get<0>(i) > (js["x2"] * qc::Am - a / 2)) && (get<0>(i) < (js["x2"] * qc::Am + a / 2))))
					get<1>(i) = -js["V0"];
			}
		}

		if (js["mode"] == "capacitor") {
			for (const auto& i : zip(x, barrier)) {
				if (((get<0>(i) > (js["x0"] * qc::Am - a / 2)) && (get<0>(i) < (js["x0"] * qc::Am + a / 2))))
					get<1>(i) = -js["V0"];
				//if (((get<0>(i) > (js["x1"] - a / 2)) && (get<0>(i) < (js["x1"] + a / 2))))
					//get<1>(i) = js["V0"];
				if (((get<0>(i) > (js["x2"] * qc::Am - a / 2)) && (get<0>(i) < (js["x2"] * qc::Am + a / 2))))
					get<1>(i) = -js["V0"];
			}
		}

		return barrier;
	}

	template <typename T>
	std::vector<T> potential_quadratic(const std::vector<T>& x) {
		auto quadratic_potential = 1e-2 * x * x;
		return quadratic_potential;
	}

	constexpr double my_pow(double x, int exp)
	{
		int sign = 1;
		if (exp < 0)
		{
			sign = -1;
			exp = -exp;
		}
		if (exp == 0)
			return x < 0 ? -1.0 : 1.0;
		double ret = x;
		while (--exp)
			ret *= x;
		return sign > 0 ? ret : 1.0 / ret;
	}

	std::string utf8_encode(std::u8string const& s)
	{
		return (const char*)(s.c_str());
	}
};


constexpr int DIM = 3;

class Maxwell {
public:
	std::array<double, DIM> ex = { 1.0, 2.0, 3.0 }; // N/C

	double phi(std::array<double, DIM> pos) {
		double phi = 0.0;
		for (int i = 0; i < DIM; i++) {
			phi += ex[i] * pos[i];
		}
		return phi;
	}

	std::array<double, DIM> a(std::array<double, DIM> pos) {
		std::array<double, DIM> a = { 0.0 };
		a[0] = -ex[1] * pos[2] + ex[2] * pos[1];
		a[1] = -ex[0] * pos[2] + ex[2] * pos[0];
		a[2] = -ex[0] * pos[1] + ex[1] * pos[0];
		return a;
	}
};

void Test_Maxwell()
{
	Maxwell maxwell;

	std::array<double, DIM> pos = { 1.7, 2.5, 7.0 }; // meters

	double phi = maxwell.phi(pos);
	std::array<double, DIM> a = maxwell.a(pos);

	std::cout << "phi = " << phi << " V" << std::endl;
	std::cout << "A = [ ";
	for (int i = 0; i < DIM; i++) {
		std::cout << a[i] << " ";
	}
	std::cout << "] V/m" << std::endl;
}

#endif //__FUNCTIONS_HPP__