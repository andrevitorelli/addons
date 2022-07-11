#ifndef SAMPLING_FUNCTIONS_H_
#define SAMPLING_FUNCTIONS_H_

#ifdef __CUDACC__
#define UNIAPI  __device__
#else
#define UNIAPI
#endif


namespace tensorflow {
namespace addons {
namespace functor {

template <ResamplingKernelType kernel_class, typename T>
class ResamplerKernelHelper {
};

template <typename T>
class ResamplerKernelHelper<ResamplingKernelType::Triangle, T> {
public:
    UNIAPI static T value(T x) {
        x = abs(x);
        return x < (T)(1.0f) ? (T)(1.0f) - x : (T)(0.0f);
    }

    UNIAPI static T derivative(T x) {
        if(x<-1.0)
            return (T)(0);
        if(x<0.0)
            return (T)(1.0);
        if(x<1.0)
            return (T)(-1.0);
        return (T)(0.0);
    }
    UNIAPI static T radius()  { return static_cast<T>(1.f); }
};

template <typename T>
class ResamplerKernelHelper<ResamplingKernelType::KeysCubic, T> {
public:
    UNIAPI static T value(T x) {
        x = abs(x);
        if (x >= 2.0f) {
            return static_cast<T>(0.0f);
        } else if (x >= 1.0f) {
            return ((static_cast<T>(-0.5f) * x + static_cast<T>(2.5f)) * x - static_cast<T>(4.0f)) * x + static_cast<T>(2.0f);
        }
        return ((static_cast<T>(1.5f) * x - static_cast<T>(2.5f)) * x) * x + static_cast<T>(1.0f);
    }

    UNIAPI static T derivative(T x) {
        if(x<-2.0) {
            return static_cast<T>(0.0);
        } else if (x<-1.0) {
            return (static_cast<T>(1.5f) * x + static_cast<T>(5.0f)) * x + static_cast<T>(4.0f);
        } else if(x<0.0) {
            return (static_cast<T>(-4.5f) * x - static_cast<T>(5.0f)) * x;
        } else if(x<1.0) {
            return (static_cast<T>(4.5f) * x - static_cast<T>(5.0f)) * x;
        } else if(x<2.0) {
            return (static_cast<T>(-1.5f) * x + static_cast<T>(5.0f)) * x - static_cast<T>(4.0f);
        }
        return static_cast<T>(0.0f);
    }
    UNIAPI static T radius() { return static_cast<T>(2.f); }
};

template <typename T>
class ResamplerKernelHelper<ResamplingKernelType::BernsteinQuintic, T> {
// Bernstein, Gary M. and Gruen, Daniel, "Resampling images in Fourier domain"
public:
    UNIAPI static T value(T x) {
        x = abs(x);
        // I'm not going fancy here, the compiler should be able to optimize the comparison sequence
        if (x <=1.0f) {
            return (((static_cast<T>(-55)*x + static_cast<T>(138))*x - static_cast<T>(95))*x*x*x + static_cast<T>(12))/static_cast<T>(12);
        } else if (x <=2.0f) {
            return (((((static_cast<T>(55)*x - static_cast<T>(414))*x + static_cast<T>(1205))*x - static_cast<T>(1680))*x + static_cast<T>(1110))*x - static_cast<T>(276))/static_cast<T>(24);
        } else if (x <= 3.0f) {
            return (((((static_cast<T>(-11)*x + static_cast<T>(138))*x - static_cast<T>(685))*x + static_cast<T>(1680))*x - static_cast<T>(2034))*x + static_cast<T>(972))/static_cast<T>(24);
        } else {
            return static_cast<T>(0);
        }
    }

    UNIAPI static T derivative(T x) {
        bool neg = signbit(x);
        x = abs(x);
        T ret;
        if (x <=1.0f) {
            ret = ((static_cast<T>(-275)*x + static_cast<T>(552))*x - static_cast<T>(285))*x*x/static_cast<T>(12);
        } else if (x <=2.0f) {
            ret = ((((static_cast<T>(275*x) - static_cast<T>(1656))*x + static_cast<T>(3615))*x - static_cast<T>(3360))*x + static_cast<T>(1110))/static_cast<T>(24);
        } else if (x <= 3.0f) {
            ret = ((((static_cast<T>(-55)*x + static_cast<T>(552))*x - static_cast<T>(2055))*x + static_cast<T>(3360))*x - static_cast<T>(2034))/static_cast<T>(24);
        } else {
            ret = static_cast<T>(0.0f);
        };
        if (neg)
              return -ret;
        else
              return ret;
    }
    UNIAPI static T radius() { return static_cast<T>(3.0f); }
};

}  // end namespace functor
}  // end namespace addons
}  // namespace tensorflow

#endif  // SAMPLING_FUNCTIONS_H_
