#ifndef __MULTICOMPLEX_HPP__
#define __MULTICOMPLEX_HPP__

#include <iostream>    
#include <vector>
#include <limits>
#include <chrono>
#include <complex>

//http://tamivox.org/eugene/multicomplex/index.html
template <typename elem, int order>
	requires ((order >= 0) && (order < 25))
class multicomplex
{

public:

	static constexpr int off{ 2 << (order - 1) };

	using one_less = multicomplex<elem, order - 1>;//recursion

	one_less Real;
	one_less Imag;

	multicomplex() = default;
	virtual ~multicomplex() = default;

	multicomplex(const one_less& r, const one_less& i) : Real(r), Imag(i) {}
	multicomplex(const elem& y) : Real(y) {}


	// assign
	template <int o_order>
	typename std::enable_if_t <(order > o_order), multicomplex<elem, order>>
operator= (multicomplex<elem, o_order> const& o) const
{
	return
	{
		o,
		Imag
	};
}

	// unary operators
	multicomplex operator+ () const
	{
		return
		{
			+Real,
			+Imag
		};
	}
	multicomplex operator~ () const
	{
		return
		{
			+Real,
			-Imag
		};
	}
	multicomplex operator- () const
	{
		return
		{
			-Real,
			-Imag
		};
	}

	// addition
	template <int o_order>
	typename std::enable_if_t <(order > o_order), multicomplex<elem, order>>
operator+ (multicomplex<elem, o_order> const& o) const
{
	return
	{
		Real + o,
		Imag
	};
}

template <int o_order>
typename std::enable_if_t < order < o_order, multicomplex<elem, o_order>>
	operator+ (multicomplex<elem, o_order> const& o) const
{
	return
	{
		*this + o.Real,
		+o.Imag
	};
}

template <int o_order>
typename std::enable_if_t <order == o_order, multicomplex<elem, order>>
operator+ (multicomplex<elem, o_order> const& o) const
{
	return
	{
		Real + o.Real,
		Imag + o.Imag
	};
}


template <typename T>
multicomplex operator+ (T const& o) const
{
	return *this + multicomplex(o);
}

template <int o_order>
typename std::enable_if_t <order == o_order, multicomplex<elem, order>>
operator+ (elem const& o)
{
	return
	{
		Real + o,
		Imag + o
	};
}

template <int o_order>
typename std::enable_if_t <(order > o_order), multicomplex<elem, order>>
operator+= (multicomplex<elem, o_order> const& o)
{
	Real += o;

	return
	{
		Real,
		Imag
	};
}

template <int o_order>
typename std::enable_if_t <order == o_order, multicomplex<elem, order>>
operator+= (multicomplex<elem, o_order> const& o)
{
	return *this = *this + o;
}

multicomplex operator+= (elem const& o)
{
	return
	{
		Real += o,
		Imag
	};
}

// prefix increment operator overloading 
multicomplex operator++ ()
{
	return *this;
}

// postfix increment operator overloading 
multicomplex operator++ (auto)
{
	return *this += 1;
}

// prefix increment operator overloading 
multicomplex operator-- ()
{
	return *this;
}

// postfix increment operator overloading 
multicomplex operator-- (auto)
{
	return *this -= 1;
}

// subtraction
template <int o_order>
typename std::enable_if_t <(order > o_order), multicomplex<elem, order>>
operator- (multicomplex<elem, o_order> const& o) const
{
	return
	{
		Real - o,
		Imag
	};
}

template <int o_order>
typename std::enable_if_t <(order < o_order), multicomplex<elem, o_order>>
	operator- (multicomplex<elem, o_order> const& o) const
{
	return
	{
		*this - o.Real,
		-o.Imag
	};
}

template <int o_order>
typename std::enable_if_t <order == o_order, multicomplex<elem, order>>
operator- (multicomplex<elem, o_order> const& o) const
{
	return
	{
		Real - o.Real,
		Imag - o.Imag
	};
}

template <int o_order>
typename std::enable_if_t <(order > o_order), multicomplex<elem, order>>
operator-= (multicomplex<elem, o_order> const& o)
{
	Real -= o;
	return
	{
		Real,
		Imag
	};
}

template <int o_order>
typename std::enable_if_t <order == o_order, multicomplex<elem, order>>
operator-= (multicomplex<elem, o_order> const& o)
{
	return *this = *this - o;
}

multicomplex operator-= (elem const& o)
{
	return
	{
		Real -= o,
		Imag
	};
}

template <typename T>
multicomplex operator- (T const& o) const
{
	return *this - multicomplex(o);
}

// multiplication
template <int o_order>
typename std::enable_if_t <(order > o_order), multicomplex<elem, order>>
operator* (multicomplex<elem, o_order> const& o) const
{
	return
	{
		Real * o,
		Imag * o
	};
}

template <int o_order>
typename std::enable_if_t < order < o_order, multicomplex<elem, o_order>>
	operator* (multicomplex<elem, o_order> const& o) const
{
	return
	{
		*this * o.Real,
		*this * o.Imag
	};
}

template <int o_order>
typename std::enable_if_t <order == o_order, multicomplex<elem, order>>
operator* (multicomplex<elem, o_order> const& o) const
{
	return
	{
		Real * o.Real - Imag * o.Imag,
		Real * o.Imag + Imag * o.Real
	};
}

template <typename  T>
multicomplex operator*= (const T& z)
{
	return *this = *this * z;
}

multicomplex operator*= (elem const& o)
{
	return *this = *this * o;
}

template <typename T>
multicomplex operator* (const T& o) const
{
	return *this * o;
}

multicomplex operator* (elem const& o) const
{
	return
	{
		Real * o,
		Imag * o
	};
}

// division
template <int o_order>
multicomplex operator/ (multicomplex<elem, o_order> const& o) const
{
	auto const new_numer{ *this * ~o };
	auto const new_denom{ o.fold() };
	return new_numer / new_denom;
}

template <typename T>
multicomplex operator/ (T const& o) const
{
	return *this / multicomplex(static_cast<elem>(o));
}

multicomplex operator/ (multicomplex<elem, 0> const& o) const
{
	return
	{
		Real / o,
		Imag / o
	};
}

multicomplex operator/ (elem const& o) const
{
	return
	{
		Real / o,
		Imag / o
	};
}

multicomplex operator/= (elem const& o)
{
	return
	{
		Real = Real / o,
		Imag = Imag / o
	};
}

template <typename  T>
multicomplex operator/= (const T& z)
{
	return *this = *this / z;
}

one_less fold() const {
	return Real * Real + Imag * Imag;
}

// miscellaneous
elem magnitude() const
{
	return std::sqrt(norm());
}
elem norm() const
{
	return Real.norm() + Imag.norm();
}
elem signr() const
{
	return Real.signr();
}
elem signi() const
{
	return Real.signi();
}
elem comp() const
{
	return Real.comp() + Imag.comp();
}

elem imag() const
{
	return Imag;
}

elem real() const
{
	return Real;
}

multicomplex clear()
{
	return *this = 0;
}

multicomplex conj()
{
	Imag = -Imag;
	return *this;
}

// selection by position

template <int x_order>
static typename
std::enable_if_t <(order > x_order), multicomplex<elem, x_order> const&>
priv_at_con(multicomplex<elem, order> const& source, int const n) {
	static constexpr int half{ 1 << (order - x_order - 1) };
	if (n < half)
		return one_less::template priv_at_con<x_order>(source.Real, n);
	else
		return one_less::template priv_at_con<x_order>(source.Imag, n - half);
}

template <int x_order>
static typename
std::enable_if_t<(order > x_order), multicomplex<elem, x_order>&>
priv_at_var(multicomplex<elem, order>& source, int const n) {
	static constexpr int half{ 1 << (order - x_order - 1) };
	if (n < half)
		return one_less::template priv_at_var<x_order>(source.Real, n);
	else
		return one_less::template priv_at_var<x_order>(source.Imag, n - half);
}

private:
	template <int x_order>
	static typename
		std::enable_if_t <order == x_order, multicomplex<elem, x_order>&>
		priv_at_var(multicomplex<elem, order>& source, int const n) {
		return source;
	}

	template <int x_order>
	static typename
		std::enable_if_t <order == x_order, multicomplex<elem, x_order> const&>
		priv_at_con(multicomplex<elem, order> const& source, int const n) {
		return source;
	}

public:
	template <int x_order>
	multicomplex<elem, x_order> const& at_con(int const n) const {
		return priv_at_con<x_order>(*this, n);
	}

	template <int x_order>
	multicomplex<elem, x_order>& at_var(int const n) {
		return priv_at_var<x_order>(*this, n);
	}

	// comparison
	template <int x_order>
	bool operator== (multicomplex<elem, x_order>& b) const
	{
		if (comp() == b.comp()) return true;
		return false;
	}

	bool operator== (const int o) const
	{
		if (Real == o) return true;
		return false;
	}

	template <int x_order>
	bool operator!= (multicomplex<elem, x_order>& b) const
	{
		if (comp() == b.comp()) return false;
		return true;
	}

	bool operator!= (const int o) const
	{
		if (Real != o) return true;
		return false;
	}

	template <int x_order>
	bool operator> (multicomplex<elem, x_order>& b) const
	{
		if (comp() > b.comp()) return true;
		return false;
	}

	bool operator> (const int o) const
	{
		if (Real > o) return true;
		return false;
	}

	template <int x_order>
	bool operator< (multicomplex<elem, x_order>& b) const
	{
		if (comp() < b.comp()) return true;
		return false;
	}

	bool operator< (const int o) const
	{
		if (Real < o) return true;
		return false;
	}

	template <int x_order>
	bool operator>= (multicomplex<elem, x_order>& b) const
	{
		if (comp() >= b.comp()) return true;
		return false;
	}

	bool operator>= (const int o) const
	{
		if (Real >= o) return true;
		return false;
	}

	template <int x_order>
	bool operator<= (multicomplex<elem, x_order>& b) const
	{
		if (comp() <= b.comp()) return true;
		return false;
	}

	bool operator<= (const int o) const
	{
		if (Real <= o) return true;
		return false;
	}

	multicomplex random(const elem&, const elem&);

	void view_i(std::ostream&) const;
	void view_j(std::ostream&) const;
	void view_j(std::ostream&, const int) const;
	void view_k(std::ostream&) const;


};

/////////////

template <typename elem>
class multicomplex<elem, 0>
{

public:

	elem Real{};
	elem Imag{};

	multicomplex() = default;
	virtual ~multicomplex() = default;

	multicomplex(const elem& r, const elem& i) : Real(r), Imag(i) {}
	multicomplex(const elem& y) : Real(y) {}

	// unary operators
	multicomplex operator+ () const
	{
		return
		{
			+Real,
			+Imag
		};
	}
	multicomplex operator~ () const
	{
		return
		{
			+Real,
			-Imag
		};
	}

	multicomplex operator- () const
	{
		return
		{
			-Real,
			-Imag
		};
	}

	// addition
	multicomplex operator+ (multicomplex const& o) const
	{
		return
		{
			Real + o.Real,
			Imag + o.Imag
		};
	}

	multicomplex operator+= (multicomplex const& o)
	{
		return *this = *this + o;
	}

	template <typename T>
	multicomplex operator+ (T const& o) const
	{
		return
		{
			Real + o,
			Imag
		};
	}

	// prefix increment operator overloading 
	multicomplex operator++ ()
	{
		return *this;
	}

	// postfix increment operator overloading 
	multicomplex operator++ (auto)
	{
		return *this += 1;
	}

	multicomplex operator+= (elem const& o)
	{
		return
		{
			Real += o,
			Imag
		};
	}

	// subtraction
	multicomplex operator- (multicomplex const& o) const
	{
		return
		{
			Real - o.Real,
			Imag - o.Imag
		};
	}

	multicomplex operator-= (multicomplex const& o)
	{
		return *this = *this - o;
	}

	// prefix increment operator overloading 
	multicomplex operator-- ()
	{
		return *this;
	}

	// postfix increment operator overloading 
	multicomplex operator-- (auto)
	{
		return *this -= 1;
	}

	// multiplication
	multicomplex operator* (multicomplex const& o) const
	{
		return
		{
			Real * o.Real - Imag * o.Imag,
			Real * o.Imag + Imag * o.Real
		};
	}

	multicomplex
		operator*= (multicomplex const& o)
	{
		return *this = *this * o;
	}

	multicomplex operator* (elem const& o) const
	{
		return
		{
			Real * o,
			Imag * o
		};
	}

	multicomplex operator*= (elem const& o)
	{
		return *this = *this * o;
	}

	// division
	multicomplex operator/ (multicomplex const& o) const
	{
		return
		{
			(Real * o.Real + Imag * o.Imag) / (o.Real * o.Real + o.Imag * o.Imag),
			(Imag * o.Real - Real * o.Imag) / (o.Real * o.Real + o.Imag * o.Imag)
		};
	}

	multicomplex operator/= (multicomplex const& o)
	{
		return *this = *this / o;
	}

	template <typename T>
	multicomplex operator/ (T const& o) const
	{
		return *this / multicomplex<elem, 0>(static_cast<elem>(o));
	}

	multicomplex operator/ (elem const& o) const
	{
		return
		{
			Real / o,
			Imag / o
		};
	}

	multicomplex operator/= (elem const& o)
	{
		return *this = *this / o;
	}

	elem fold()
	{
		return Real * Real + Imag * Imag;
	}

	// miscellaneous
	elem signr() const
	{
		return Real;
	}
	elem signi() const
	{
		return Imag;
	}
	elem magnitude() const
	{
		return std::sqrt(norm());
	}
	elem norm() const
	{
		return Real * Real + Imag * Imag;
	}
	elem comp() const
	{
		return Real * Imag + Real + Imag;
	}

	multicomplex clear()
	{
		return *this = 0;
	}

	multicomplex conj()
	{
		Imag = -Imag;
		return *this;
	}
	
	elem imag() const
	{
		return Imag;
	}

	elem real() const
	{
		return Real;
	}

	// selection by position
	template <int x_order>
	static typename
		std::enable_if <0 == x_order, multicomplex<elem, x_order> const&>::type
		priv_at_con(multicomplex<elem, 0> const& source, int const n) {
		return source;
	}

	template <int x_order>
	static typename
		std::enable_if <0 == x_order, multicomplex<elem, x_order>&>::type
		priv_at_var(multicomplex<elem, 0>& source, int const n) {
		return source;
	}
private:
	template <int x_order>
		requires (x_order == 0)
	multicomplex<elem, x_order> const& at_con(int const n) const {
		return priv_at_con<x_order>(*this, n);
	}

	template <int x_order>
		requires (x_order == 0)
	multicomplex<elem, x_order>& at_var(int const n) {
		return priv_at_var<x_order>(*this, n);
	}
public:
	// comparison
	template <int x_order>
	bool operator== (const multicomplex<elem, x_order>& b) const
	{
		if (comp() == b.comp()) return true;
		return false;
	}

	bool operator== (const elem& o) const
	{
		if (comp() == multicomplex<elem, 0>(static_cast<elem>(o)).comp()) return true;
		return false;
	}

	template <typename T>
	bool operator== (const T o) const
	{
		if (Real == o) return true;
		return false;
	}

	template <int x_order>
	bool operator!= (const multicomplex<elem, x_order>& b) const
	{
		if (comp() == b.comp()) return false;
		return true;
	}

	bool operator!= (const elem& o) const
	{
		if (comp() == multicomplex<elem, 0>(static_cast<elem>(o)).comp()) return false;
		return true;
	}

	bool operator!= (const int o) const
	{
		if (Real != o) return true;
		return false;
	}

	template <int x_order>
	bool operator> (const multicomplex<elem, x_order>& b) const
	{
		if (comp() > b.comp()) return true;
		return false;
	}

	bool operator> (const int o) const
	{
		if (Real > o) return true;
		return false;
	}

	bool operator> (const elem& o) const
	{
		if (comp() > multicomplex<elem, 0>(static_cast<elem>(o)).comp()) return true;
		return false;
	}

	bool operator< (const int o) const
	{
		if (Real < o) return true;
		return false;
	}

	template <int x_order>
	bool operator< (const multicomplex<elem, x_order>& b) const
	{
		if (comp() < b.comp()) return true;
		return false;
	}

	bool operator< (const elem& o) const
	{
		if (comp() < multicomplex<elem, 0>(static_cast<elem>(o)).comp()) return true;
		return false;
	}

	template <int x_order>
	bool operator>= (const multicomplex<elem, x_order>& b) const
	{
		if (comp() >= b.comp()) return true;
		return false;
	}

	bool operator>= (const elem& o) const
	{
		if (comp() >= multicomplex<elem, 0>(static_cast<elem>(o)).comp()) return true;
		return false;
	}

	bool operator>= (const int o) const
	{
		if (Real >= o) return true;
		return false;
	}

	template <int x_order>
	bool operator<= (const multicomplex<elem, x_order>& b) const
	{
		if (comp() <= b.comp()) return true;
		return false;
	}

	bool operator<= (const int o) const
	{
		if (Real <= o) return true;
		return false;
	}

	bool operator<= (const elem& o) const
	{
		if (comp() <= multicomplex<elem, 0>(static_cast<elem>(o)).comp()) return true;
		return false;
	}

	void view_i(std::ostream&) const;
	void view_j(std::ostream&) const;
	void view_j(std::ostream&, const int) const;
	void view_k(std::ostream&) const;

	multicomplex random(const elem&, const elem&);

};

/////////////////////////////////////////

template <typename old_elem, typename new_elem, int order>
multicomplex<new_elem, order> convert_elem(multicomplex<old_elem, order> const& o)
{
	return multicomplex<new_elem, order> {
		convert_elem<old_elem, new_elem, order - 1>(o.Real),
			convert_elem<old_elem, new_elem, order - 1>(o.Imag)
	};
}


//typedefs
typedef double REAL;
typedef multicomplex<REAL, 0 > MX0;
typedef multicomplex<REAL, 1 > MX1;
typedef multicomplex<REAL, 2 > MX2;
typedef multicomplex<REAL, 3 > MX3;
typedef multicomplex<REAL, 4 > MX4;
typedef multicomplex<REAL, 5 > MX5;
typedef multicomplex<REAL, 6 > MX6;
typedef multicomplex<REAL, 7 > MX7;
typedef multicomplex<REAL, 8 > MX8;
typedef multicomplex<REAL, 9 > MX9;
typedef multicomplex<REAL, 10 > MX10;
typedef multicomplex<REAL, 11 > MX11;
typedef multicomplex<REAL, 12 > MX12;
typedef multicomplex<REAL, 13 > MX13;
typedef multicomplex<REAL, 14 > MX14;
typedef multicomplex<REAL, 15 > MX15;
typedef multicomplex<REAL, 16 > MX16;


constexpr const REAL pi = 3.141592653589793238462643383279502884197169399375105820974;
constexpr const REAL two_pi = 6.2831853071795864769252867665590057683943387987502116419498891846;
constexpr const REAL half_pi = 1.570796326794896619231321691639751442098584699687552910487;
constexpr const REAL euler = 0.577215664901532860606512090082402431042159335939923598805;
constexpr const REAL half = 0.5;
constexpr const REAL quarter = 0.25;
constexpr const REAL root_two_pi = 1.772453850905516027298167483341145182797549456122387128213;
constexpr const REAL root_two = 1.4142135623730950488016887242096980785696718753769480731766797379;
constexpr const REAL PHI = 1.6180339887498948482045868343656381177203091798057628621354486227;

constexpr const REAL lambda = std::numeric_limits<REAL>::epsilon();


//constexpr const REAL lambda = 1e-5;
typedef std::vector<REAL> Vec;

//////////////////
#include <mxws.hpp>
//////////////////
#include <flat.hpp>
//////////////////
#include <operators.hpp>
//////////////////
#include <basic.hpp>
//////////////////
#include <additional_functions.hpp>
//////////////////
#include <derivative.hpp>
//////////////////
#include <root.hpp>
//////////////////
#include <elementry_functions.hpp>
//////////////////
#include <integrator.hpp>
//////////////////
#include <test.hpp>
//////////////////
#include <linked_list.hpp>
//////////////////
#include <polynomals.hpp>

template <typename elem, int order>
bool operator< (const int a, const multicomplex<elem, order>& b)
{
	if (a < b.Real) return true;
	return false;
}

template <typename elem, int order>
bool operator<= (const int a, const multicomplex<elem, order>& b)
{
	if (a <= b.Real) return true;
	return false;
}

template <typename elem, int order>
bool operator> (const int a, const multicomplex<elem, order>& b)
{
	if (a > b.Real) return true;
	return false;
}

template <typename elem, int order>
bool operator>= (const int a, const multicomplex<elem, order>& b)
{
	if (a >= b.Real) return true;
	return false;
}

template <typename elem, int order>
bool operator== (const int a, const multicomplex<elem, order>& b)
{
	if (a == b.Real) return true;
	return false;
}

template <typename elem, int order>
bool operator!= (const int a, const multicomplex<elem, order>& b)
{
	if (a != b.Real) return true;
	return false;
}


template <typename elem, int order>
	requires ((order >= 0) && (order < 25))
multicomplex<elem, order> multicomplex<elem, order>::random(const elem& lower, const elem& upper)
{
	mxws <uint32_t>rng;
	flat a = *this;

	for (int j = 0; j < a.size(); j++)
		a.at(j) = rng(lower, upper);

	*this = a;
	return *this;
}

template <typename elem>
multicomplex<elem, 0> multicomplex<elem, 0>::random(const elem& lower, const elem& upper)
{
	mxws <uint32_t>rng;
	flat a = *this;

	for (int j = 0; j < a.size(); j++)
		a.at(j) = rng(lower, upper);

	*this = a;
	return *this;
}

//cout
template <typename elem, int order>
	requires ((order >= 0) && (order < 25))
void multicomplex<elem, order>::view_i
(
	std::ostream& o
) const
{
	o << "(" << Real << " " << "+ " << Imag << "*i" << order + 1 << ")";
}

template <typename elem>
void multicomplex<elem, 0>::view_i
(
	std::ostream& o
) const
{
	o << "(";
	if (Real >= 0)
		o << "+ " << +Real;
	else
		o << "- " << -Real;
	if (Imag >= 0)
		o << " + " << +Imag;
	else
		o << " - " << -Imag;
	o << "*i1" << ")";
}

template <typename elem, int order>
	requires ((order >= 0) && (order < 25))
void multicomplex<elem, order>::view_j
(
	std::ostream& o
) const
{
	o << "(";
	view_j(o, 0);
	o << ")";
}

template <typename elem, int order>
	requires ((order >= 0) && (order < 25))
void multicomplex<elem, order>::view_j
(
	std::ostream& o,
	int const sum
) const
{
	Real.view_j(o, sum);
	o << " ";
	Imag.view_j(o, sum + off);
}

template <typename elem>
void multicomplex<elem, 0>::view_j
(
	std::ostream& o, int const sum
) const
{
	if (Real >= 0)
		o << "+ " << +Real;
	else
		o << "- " << -Real;
	o << "*j" << sum;

	if (Imag >= 0)
		o << " + " << +Imag;
	else
		o << " - " << -Imag;
	o << "*j" << sum + 1;
}

template <typename elem>
void multicomplex<elem, 0>::view_j
(
	std::ostream& o
) const
{
	o << "(";
	view_j(o, 0);
	o << ")";
}

template <typename elem, int order>
	requires ((order >= 0) && (order < 25))
void multicomplex<elem, order>::view_k
(
	std::ostream& o
) const
{
	o << "[";
	Real.view_k(o);
	o << " ";
	Imag.view_k(o);
	o << "]";
}

template <typename elem>
void multicomplex<elem, 0>::view_k
(
	std::ostream& o
) const
{
	o << "[";
	o << Real << " ";
	o << Imag;
	o << "]";
}

template <typename elem, int order>
std::ostream& operator <<
(
	std::ostream& o,
	multicomplex<elem, order> const& mc
	)
{
	mc.view_j(o);
	return o;
}

template <typename elem>
std::ostream& operator <<
(
	std::ostream& o,
	multicomplex<elem, 0> const& mc
	)
{
	mc.view_i(o);
	return o;
}

#endif //__MULTICOMPLEX_HPP__
