//Copyright (c) 2015 Zachary Kann
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

// ---
// Author: Zachary Kann

#include "z_stat.hpp"

double LinearFitSlope(const double x_interval, const arma::rowvec& y,
                      const int startfit, const int endfit) {
  double slope, xsum = 0.0, ysum = 0.0, xysum = 0.0, x2sum = 0.0;
  for(int i1=startfit; i1<endfit; i1++) {
    xsum += i1*x_interval;
    ysum += y(i1);
    xysum += i1*x_interval*y(i1);
    x2sum += i1*x_interval*i1*x_interval;
  }
  slope =
      ((endfit-startfit)*xysum - xsum*ysum)/((endfit-startfit)*x2sum-xsum*xsum);
  return slope;
}

// This algorithm improves on the standard least squares exponential fit by
// removing the bias towards small y values.
// See Weisstein, Eric W. "Least Squares Fitting--Exponential."
// From MathWorld--A Wolfram Web Resource.
// http://mathworld.wolfram.com/LeastSquaresFittingExponential.html
double ExponentialFitSlope(const double x_interval, const arma::irowvec& y,
                           const int start_fit, const int end_fit) {
  double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0, sum5 = 0.0;
  for (int i = start_fit; i < end_fit; i++) {
    double w = log(y[i]);
    sum1 += y[i];                    //y
    sum2 += x_interval*i*y[i]*w; //x*y*ln(y)
    sum3 += x_interval*i*y[i];   //x*y
    sum4 += y[i]*w;                  //y*ln(y)
    sum5 += x_interval*i*x_interval*i*y[i]; //x^2*y
  }
  double slope = (sum1*sum2 - sum3*sum4)/(sum1*sum5-sum3*sum3);
  return slope;
}
