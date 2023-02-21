#ifndef __FFTOPNMP_H__
#define __FFTOPNMP_H__

#include <cmath>
#include <cstdlib>
#include <complex>
#include <valarray>
#include <omp.h>

//https://people.sc.fsu.edu/~jburkardt/cpp_src/fft_openmp/fft_openmp.cpp
// 
//****************************************************************************80

void cffti(int n, double w[])

//****************************************************************************80
//
//  Purpose:
//
//    CFFTI sets up sine and cosine tables needed for the FFT calculation.
//
//  Modified:
//
//    20 March 2009
//
//  Author:
//
//    Original C version by Wesley Petersen.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Wesley Petersen, Peter Arbenz, 
//    Introduction to Parallel Computing - A practical guide with examples in C,
//    Oxford University Press,
//    ISBN: 0-19-851576-6,
//    LC: QA76.58.P47.
//
//  Parameters:
//
//    Input, int N, the size of the array to be transformed.
//
//    Output, double W[N], a table of sines and cosines.
//
{
  double arg;
  double aw;
  int i;
  int n2;
  
  n2 = n / 2;
  aw = 2 * std::numbers::pi / n;

# pragma omp parallel \
    shared ( aw, n, w ) \
    private ( arg, i )

# pragma omp for nowait

  for (i = 0; i < n2; i++)
  {
    arg = aw * ((double)i);
    w[i * 2 + 0] = std::cos(arg);
    w[i * 2 + 1] = std::sin(arg);
  }
  return;
}

//****************************************************************************80

void step(int n, int mj, double a[], double b[], std::vector<double>& c,
  double d[], const std::vector<double>& w, double sgn)

  //****************************************************************************80
  //
  //  Purpose:
  //
  //    STEP carries out one step of the workspace version of CFFT2.
  //
  //  Modified:
  //
  //    20 March 2009
  //
  //  Author:
  //
  //    Original C version by Wesley Petersen.
  //    C++ version by John Burkardt.
  //
  //  Reference:
  //
  //    Wesley Petersen, Peter Arbenz, 
  //    Introduction to Parallel Computing - A practical guide with examples in C,
  //    Oxford University Press,
  //    ISBN: 0-19-851576-6,
  //    LC: QA76.58.P47.
  //
{
  double ambr;
  double ambu;
  int j;
  int ja;
  int jb;
  int jc;
  int jd;
  int jw;
  int k;
  int lj;
  int mj2;
  double wjw[2];

  mj2 = 2 * mj;
  lj = n / mj2;

# pragma omp parallel \
    shared ( a, b, c, d, lj, mj, mj2, sgn, w ) \
    private ( ambr, ambu, j, ja, jb, jc, jd, jw, k, wjw )

# pragma omp for nowait

  for (j = 0; j < lj; j++)
  {
    jw = j * mj;
    ja = jw;
    jb = ja;
    jc = j * mj2;
    jd = jc;

    wjw[0] = w[jw * 2];
    wjw[1] = w[jw * 2 + 1];

    if (sgn < 0)
    {
      wjw[1] = -wjw[1];
    }

    for (k = 0; k < mj; k++)
    {
      c[(jc + k) * 2] = a[(ja + k) * 2] + b[(jb + k) * 2];
      c[(jc + k) * 2 + 1] = a[(ja + k) * 2 + 1] + b[(jb + k) * 2 + 1];

      ambr = a[(ja + k) * 2] - b[(jb + k) * 2];
      ambu = a[(ja + k) * 2 + 1] - b[(jb + k) * 2 + 1];

      d[(jd + k) * 2] = wjw[0] * ambr - wjw[1] * ambu;
      d[(jd + k) * 2 + 1] = wjw[1] * ambr + wjw[0] * ambu;
    }
  }
  return;
}


//****************************************************************************80

void cfft2(int n, std::vector<double>& x, std::vector<double>& y, std::vector<double>& w, double sgn)

//****************************************************************************80
//
//  Purpose:
//
//    CFFT2 performs a complex Fast Fourier Transform.
//
//  Modified:
//
//    20 March 2009
//
//  Author:
//
//    Original C version by Wesley Petersen.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Wesley Petersen, Peter Arbenz, 
//    Introduction to Parallel Computing - A practical guide with examples in C,
//    Oxford University Press,
//    ISBN: 0-19-851576-6,
//    LC: QA76.58.P47.
//
//  Parameters:
//
//    Input, int N, the size of the array to be transformed.
//
//    Input/output, double X[2*N], the data to be transformed.  
//    On output, the contents of X have been overwritten by work information.
//
//    Output, double Y[2*N], the forward or backward FFT of X.
//
//    Input, double W[N], a table of sines and cosines.
//
//    Input, double SGN, is +1 for a "forward" FFT and -1 for a "backward" FFT.
//
{
  int j;
  int m;
  int mj;
  int tgle;

  m = int(log2(n));
  mj = 1;
  //
  //  Toggling switch for work array.
  //
  tgle = 1;
  step(n, mj, x.data(), x.data() + n, y, y.data() + mj * 2, w, sgn);

  if (n == 2)
  {
    return;
  }

  for (j = 0; j < m - 2; j++)
  {
    mj *= 2;
    if (tgle)
    {
      step(n, mj, y.data(), y.data() + n, x, x.data() + mj * 2, w, sgn);
      tgle = 0;
    }
    else
    {
      step(n, mj, x.data(), x.data() + n, y, y.data() + mj * 2, w, sgn);
      tgle = 1;
    }
  }
  //
  //  Last pass thru data: move y to x if needed 
  //
  if (tgle)
  {
    std::ranges::copy(y, x.begin());
  }

  mj = n / 2;
  step(n, mj, x.data(), x.data() + n, y, y.data() + mj * 2, w, sgn);

  return;
}


#endif //  __FFTOPNMP_H__