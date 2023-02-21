
#ifndef __VECTOR_OPERATORS_HPP__
#define __VECTOR_OPERATORS_HPP__

template<typename ErrorType>
void throwError(const std::string& file, const std::string& function, std::uint32_t line, const std::string& msg = "")
{
	std::string errMsg = "File: " + file + "\n\tFunction: " + function + "\n\tLine: " + std::to_string(line) +
		"\n\tError: " + msg;
	std::cerr << errMsg;
	throw ErrorType(errMsg);
}

#define THROW_INVALID_ARGUMENT_ERROR(msg) \
    throwError<std::invalid_argument>(__FILE__, __func__, __LINE__, msg)
#define THROW_RUNTIME_ERROR(msg) throwError<std::runtime_error>(__FILE__, __func__, __LINE__, msg)

template<class... Args>
auto vec(size_t n, Args&&... args) {
	if constexpr (sizeof...(args) == 1)
		return std::vector(n, args...);
	else
		return std::vector(n, vec(args...));
}

template <typename T>
inline const std::vector<T> operator*
(
	const std::vector<T>& a,
	const std::vector<T>& b
	)
{
	std::vector<T> v = a;

	for (auto i = 0; i < a.size(); i++)
		v[i] *= b[i];

	return v;
}

template <typename T>
inline const std::vector<Complex> operator*
(
	const Complex& b,
	const std::vector<T>& a
	)
{
	std::vector<Complex> v(a.size());

	for (auto i = 0; i < a.size(); i++)
		v[i] = b * a[i];

	return v;
}

template <typename T>
inline const std::vector<Complex> operator*
(
	const std::vector<T>& a,
	const Complex& b

	)
{
	std::vector<Complex> v(a.size());

	for (auto i = 0; i < a.size(); i++)
		v[i] = b * a[i];

	return v;
}

template <typename A, typename T>
inline const std::vector<T> operator*
(
	const std::vector<T>& a,
	const A& b
	)
{
	std::vector<T> v = a;

	for (auto i = 0; i < a.size(); i++)
		v[i] *= b;

	return v;
}

template <typename T>
inline const std::vector < std::vector<T>> operator*
(
	const T& b,
	const std::vector < std::vector<T>>& a
	)
{
	std::vector < std::vector<T>> v = a;

	for (auto& i : v)
		for (auto& k : i)
			k = k * b;

	return v;
}

template <typename T>
inline std::vector<T> operator*=
(
	const T& a,
	std::vector<T>& b
	)
{
	for (auto i = 0; i < b.size(); i++)
		b[i] *= a;
	return b;
}

template <typename T>
inline const std::vector<T> operator*=
(
	std::vector<T>& a,
	const std::vector<T>& b
	)
{
	a = a * b;
	return a;
}

template <typename T>
inline std::vector<T> operator*=
(
	std::vector<T>& b,
	const T& a
	)
{
	for (auto i = 0; i < b.size(); i++)
		b[i] *= a;
	return b;
}

template <typename T>
inline const std::vector<T> operator+
(
	const std::vector<T>& a,
	const std::vector<T>& b
	)
{
	std::vector<T> v = a;

	for (auto i = 0; i < a.size(); i++)
		v[i] += b[i];

	return v;
}

template <typename T>
inline const std::vector<T> operator+
(
	const T& a,
	const std::vector<T>& b
	)
{
	std::vector<T> v = b;

	for (auto i = 0; i < b.size(); i++)
		v[i] += a;

	return v;
}

template <typename T>
inline const std::vector<T> operator+
(
	const std::vector<T>& a,
	const T& b
	)
{
	std::vector<T> v = a;

	for (auto i = 0; i < a.size(); i++)
		v[i] += b;

	return v;
}

template <typename A, typename T>
inline const std::vector<T> operator+
(
	const std::vector<T>& a,
	const A& b
	)
{
	std::vector<T> v = a;

	for (auto i = 0; i < a.size(); i++)
		v[i] += b;

	return v;
}


template <typename T>
inline const std::vector<T> operator+=
(
	std::vector<T>& a,
	const std::vector<T>& b
	)
{
	for (auto i = 0; i < a.size(); i++)
		a[i] += b[i];

	return a;
}

template <typename T>
inline const std::vector < std::vector<T>> operator+=
(
	const std::vector < std::vector<T>>& a,
	const std::vector<T>& b
	)
{
	std::vector < std::vector<T>> v = a;
	for (auto& i : v)
			i = i + b;

	return v;
}


template <typename T>
inline const std::vector<T> operator-
(
	const std::vector<T>& a,
	const std::vector<T>& b
	)
{
	std::vector<T> v = a;

	for (auto i = 0; i < a.size(); i++)
		v[i] -= b[i];

	return v;
}

template <typename T>
inline const std::vector < std::vector<T>> operator-
(
	const std::vector < std::vector<T>>& a,
	const std::vector < std::vector<T>>& b
	)
{
	std::vector < std::vector<T>> v = a;

	for (auto i = 0; i < a.size(); i++)
		v[i] = v[i] - b[i];

	return v;
}

template <typename T>
inline const std::vector<T> operator-
(
	const T& a,
	const std::vector<T>& b
	)
{
	std::vector<T> v = b;

	for (auto i = 0; i < b.size(); i++)
		v[i] -= a;

	return v;
}

template <typename T>
inline const std::vector<T> operator-
(
	const std::vector<T>& a
	)
{
	std::vector<T> v = a;

	for (auto i = 0; i < a.size(); i++)
		v[i] = -v[i];

	return v;
}

template <typename T>
inline const std::vector<T> operator-
(
	const std::vector<T>& a,
	const T& b
	)
{
	std::vector<T> v = a;

	for (auto i = 0; i < a.size(); i++)
		v[i] -= b;

	return v;
}

template <typename T, typename B>
inline const std::vector < std::vector<T>> operator-
(
	const std::vector < std::vector<T>>& a,
	const B& b
	)
{
	std::vector < std::vector<T>> v = a;

	for (auto i = 0; i < a.size(); i++)
		v[i] = v[i] - b;

	return v;
}

template <typename T>
inline const std::vector<T> operator/
(
	const std::vector<T>& a,
	const std::vector<T>& b
	)
{
	std::vector<T> v = a;

	for (auto i = 0; i < a.size(); i++)
		v[i] /= b[i];

	return v;
}

template <typename A, typename T>
inline const std::vector < std::vector<T>> operator/
(
	const A& a,
	const std::vector < std::vector<T>>& b
	)
{
	std::vector < std::vector<T>> v = b;

	for (auto y = 0; y < b.size(); y++)
		for (auto i = 0; i < b[0].size(); i++)
		v[y][i] = T(a) / v[y][i];

	return v;
}

template <typename A, typename T>
inline const std::vector<T> operator/
(
	const A& a,
	const std::vector<T>& b
	)
{
	std::vector<T> v = b;

	for (auto i = 0; i < b.size(); i++)
		v[i] /= a;

	return v;
}

template <typename A, typename T>
inline const std::vector<T> operator/
(
	std::vector<T> a,
	const A& b
	)
{
	int i = 0;
# pragma omp parallel shared(a) private(i)
	{
# pragma omp for nowait      
		for (i = 0; i < a.size(); i++)
			a[i] /= b;
	}
	return a;
}

template <typename A, typename T>
inline const std::vector<T> operator/=
(
	std::vector<T>& a,
	const A& b
	)
{
	a = a / b;
	return a;
}

template <typename T>
inline std::vector<T> operator/=
(
	const T& a,
	std::vector<T>& b
	)
{
	for (auto i = 0; i < b.size(); i++)
		b[i] /= a;
	return b;
}

template <typename T>
inline std::vector<T> operator/=
(
	std::vector<T>& b,
	const T& a
	)
{
	for (auto i = 0; i < b.size(); i++)
		b[i] /= a;
	return b;
}

template <typename T>
inline const std::vector<T> exp
(
	const std::vector<T>& a
)
{
	std::vector<T> v = a;

	for (auto i = 0; i < a.size(); i++)
		v[i] = exp(a[i]);

	return v;
}

template <typename T>
inline const std::vector<T> pow
(
	const std::vector<T>& a,
	const T& b
)
{
	std::vector<T> v = a;

	for (auto i = 0; i < a.size(); i++)
		v[i] = pow(a[i], b);

	return v;
}

template <typename A, typename T>
inline const std::vector<T> pow
(
	const std::vector<T>& a,
	const A& b
)
{
	std::vector<T> v = a;

	for (auto i = 0; i < a.size(); i++)
		v[i] = pow(a[i], b);

	return v;
}

template<typename dtype>
inline const std::vector < std::vector<dtype>>operator<<
(
	std::ostream& o,
	const std::vector < std::vector<dtype>>& v
	)
{
	int t = 0;

	for (auto& i : v) {

		for (auto& k : i) {
			o << std::scientific << std::setprecision(8) << k << " ";
			if ((t++) > 2) { t = 0; o << std::endl; }
		}
	}

	return v;
}

template<typename dtype>
inline const std::vector<dtype>operator<<
(
	std::ostream& o,
	const std::vector<dtype>& v
	)
{
	for (auto& i : v)
		o << std::scientific << std::setprecision(8) << i << " ";

	return v;
}

template <typename T>
const std::vector < std::vector<T>> pow
(
	std::vector < std::vector<T>> v, 
	int e
)
{
	for (auto& i : v)
		for (auto& k : i)
			k = pow(k, e);
	return v;
}

template <typename T>
const std::vector<T> abs(std::vector<T> v)
{
	for (auto& i : v)
		i = abs(i);
	return v;
}

template <typename T, typename B>
inline const std::vector<bool> operator<
	(
		const std::vector<T>& b,
		const B& a
		)
{
	std::vector<bool> v(b.size(), false);

	for (auto i = 0; i < b.size(); i++)
		if (b[i] < T(a))
			v[i] = true;

	return v;
}

template <typename T, typename B>
inline const std::vector < std::vector<bool>> operator<
	(
		const std::vector < std::vector<T>>& b,
		const B& a
		)
{
	std::vector<bool> v1(b[0].size(), false);
	std::vector < std::vector<bool>> v(b.size(), v1);

	for (auto k = 0; k < b.size(); k++)
		for (auto i = 0; i < b[0].size(); i++)
		if (b[k][i] < T(a))
			v[k][i] = true;

	return v;
}

template <typename T, typename B>
inline const std::vector<bool> operator>
	(
		const std::vector<T>& b,
		const B& a
		)
{
	std::vector<bool> v(b.size(), false);

	for (auto i = 0; i < v.size(); i++)
		if (b[i] > T(a))
			v[i] = true;

	return v;
}

template <typename T, typename B>
inline const std::vector<bool> operator<=
	(
		const std::vector<T>& b,
		const B& a
		)
{
	std::vector<bool> v(b.size(), false);

	for (auto i = 0; i < b.size(); i++)
		if (b[i] <= T(a))
			v[i] = true;

	return v;
}

template <typename T, typename B>
inline const std::vector<bool> operator>=
	(
		const std::vector<T>& b,
		const B& a
		)
{
	std::vector<bool> v(b.size(), false);

	for (auto i = 0; i < b.size(); i++)
		if (b[i] < T(a))
			v[i] >= true;

	return v;
}

template <typename T>
inline const std::vector<bool> operator==
	(
		const std::vector<T>& b,
		const int& a
		)
{
	std::vector<bool> v(b.size(), false);

	for (auto i = 0; i < b.size(); i++)
		if (b[i] == T(a))
			v[i] = true;

	return v;
}

template <typename T>
inline const std::vector<bool> operator==
(
	const std::vector<T>& b,
	const double& a
	)
{
	std::vector<bool> v(b.size(), false);

	for (auto i = 0; i < b.size(); i++)
		if (b[i] == T(a))
			v[i] = true;

	return v;
}

template <typename T, typename B>
inline const std::vector<bool> operator!=
	(
		const std::vector<T>& b,
		const B& a
		)
{
	std::vector<bool> v(b.size(), false);

	for (auto i = 0; i < b.size(); i++)
		if (b[i] != T(a))
			v[i] = true;

	return v;
}

inline const std::vector<bool> operator|
(
	const std::vector<bool>& a,
	const std::vector<bool>& b
	)
{
	std::vector<bool> v(b.size(), false);

	for (auto i = 0; i < b.size(); i++)
		if (a[i] | b[i])
			v[i] = true;

	return v;
}

inline const std::vector<bool> operator==
(
	const std::vector<bool>& a,
	const std::vector<bool>& b
	)
{
	std::vector<bool> v(b.size(), false);

	for (auto i = 0; i < b.size(); i++)
		if (a[i] == b[i])
			v[i] = true;

	return v;
}

inline const std::vector<bool> operator&
(
	const std::vector<bool>& a,
	const std::vector<bool>& b
	)
{
	std::vector<bool> v(b.size(), false);

	for (auto i = 0; i < b.size(); i++)
		if (a[i] & b[i])
			v[i] = true;

	return v;
}


inline const std::vector < std::vector<bool>> operator&
	(
		const std::vector < std::vector<bool>>& a,
		const std::vector < std::vector<bool>>& b
		)
{
	std::vector<bool> v1(b[0].size(), false);
	std::vector < std::vector<bool>> v(b.size(), v1);

	for (auto k = 0; k < b.size(); k++)
		for (auto i = 0; i < b[0].size(); i++)
			if (a[k][i] & b[k][i])
				v[k][i] = true;

	return v;
}

inline const std::vector < std::vector<bool>> operator|
(
	const std::vector < std::vector<bool>>& a,
	const std::vector < std::vector<bool>>& b
	)
{
	std::vector<bool> v1(b[0].size(), false);
	std::vector < std::vector<bool>> v(b.size(), v1);

	for (auto k = 0; k < b.size(); k++)
		for (auto i = 0; i < b[0].size(); i++)
			if (a[k][i] | b[k][i])
				v[k][i] = true;

	return v;
}

inline const std::vector < std::vector<bool>> operator==
(
	const std::vector < std::vector<bool>>& a,
	const std::vector < std::vector<bool>>& b
	)
{
	std::vector<bool> v1(b[0].size(), false);
	std::vector < std::vector<bool>> v(b.size(), v1);

	for (auto k = 0; k < b.size(); k++)
		for (auto i = 0; i < b[0].size(); i++)
			if (a[k][i] == b[k][i])
				v[k][i] = true;

	return v;
}

template<typename dtype>
std::vector<dtype> where(const std::vector<bool>& inMask, const std::vector<dtype>& inA,
	const std::vector<dtype>& inB)
{
	const auto shapeMask = inMask.size();
	const auto shapeA = inA.size();
	if (shapeA != inB.shape())
	{
		THROW_INVALID_ARGUMENT_ERROR("input inA and inB must be the same shapes.");
	}

	if (shapeMask != shapeA)
	{
		THROW_INVALID_ARGUMENT_ERROR("input inMask must be the same shape as the input arrays.");
	}

	auto outArray = std::vector<dtype>(shapeMask);

	auto idx = 0;
	for (auto maskValue : inMask)
	{
		if (maskValue)
		{
			outArray[idx] = inA[idx];
		}
		else
		{
			outArray[idx] = inB[idx];
		}
		++idx;
	}

	return outArray;
}

template<typename dtype, typename dtype2>
std::vector<dtype> where(const std::vector<bool>& inMask, const std::vector<dtype>& inA, dtype2 inB)
{
	const auto shapeMask = inMask.size();
	const auto shapeA = inA.size();
	if (shapeMask != shapeA)
	{
		THROW_INVALID_ARGUMENT_ERROR("input inMask must be the same shape as the input arrays.");
	}

	auto outArray = std::vector<dtype>(shapeMask);

	auto idx = 0;
	for (auto maskValue : inMask)
	{
		if (maskValue)
		{
			outArray[idx] = inA[idx];
		}
		else
		{
			outArray[idx] = inB;
		}
		++idx;
	}

	return outArray;
}

template<typename dtype, typename dtype2>
std::vector<dtype> where(const std::vector<bool>& inMask, dtype2 inA, const std::vector<dtype>& inB)
{
	const auto shapeMask = inMask.size();
	const auto shapeB = inB.size();
	if (shapeMask != shapeB)
	{
		THROW_INVALID_ARGUMENT_ERROR("input inMask must be the same shape as the input arrays.");
	}

	auto outArray = std::vector<dtype>(shapeMask);

	auto idx = 0;
	for (auto maskValue : inMask)
	{
		if (maskValue)
		{
			outArray[idx] = inA;
		}
		else
		{
			outArray[idx] = inB[idx];
		}
		++idx;
	}

	return outArray;
}

/// <summary>
/// 
/// </summary>
/// <typeparam name="dtype"></typeparam>
/// <typeparam name="dtype2"></typeparam>
/// <param name="inMask"></param>
/// <param name="inA"></param>
/// <param name="inB"></param>
/// <returns></returns>
template<typename dtype, typename dtype2>
std::vector < std::vector<dtype>> where(const std::vector < std::vector<bool>>& inMask, dtype2 inA,
	const std::vector < std::vector<dtype>>& inB)
{
	const auto shapeMask = inMask.size();
	const auto shapeB = inB.size();
	if (shapeMask != shapeB)
	{
		THROW_INVALID_ARGUMENT_ERROR("input inMask must be the same shape as the input arrays.");
	}

	std::vector<dtype> v(inMask[0].size());
	auto outArray = std::vector < std::vector<dtype>>(shapeMask, v);

	for (auto y = 0; y < inMask.size(); y++) {
	
		for (auto x = 0; x < inMask[0].size(); x++)
		{
			if (inMask[y][x])
			{
				outArray[y][x] = inA;
			}
			else
			{
				outArray[y][x] = inB[y][x];
			}
		}
	
	}
	return outArray;
}

template<typename dtype>
std::vector<dtype> where(const std::vector<bool>& inMask, dtype inA, dtype inB)
{
	auto outArray = std::vector<dtype>(inMask.size());

	auto idx = 0;
	for (auto maskValue : inMask)
	{
		if (maskValue)
		{
			outArray[idx] = inA;
		}
		else
		{
			outArray[idx] = inB;
		}
		++idx;
	}

	return outArray;
}

#endif // __VECTOR_OPERATORS_HPP__

