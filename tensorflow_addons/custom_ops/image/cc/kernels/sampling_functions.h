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
        x = std::abs(x);
        return x < 1.0f ? 1.0f - x : 0.0f;
    }

    UNIAPI static T derivative(T x) {
        if(x<-1.0)
            return 0;
        if(x<0.0)
            return 1.0;
        if(x<1.0)
            return -1.0;
        return 0.0;
    }
    UNIAPI static T radius()  { return static_cast<T>(1.f); }
};

template <typename T>
class ResamplerKernelHelper<ResamplingKernelType::KeysCubic, T> {
public:
    UNIAPI static T value(T x) {
        x = std::abs(x);
        if (x >= 2.0f) {
            return 0.0f;
        } else if (x >= 1.0f) {
            return ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
        }
        return ((1.5f * x - 2.5f) * x) * x + 1.0f;
    }

    UNIAPI static T derivative(T x) {
        if(x<-2.0) {
            return 0.0;
        } else if (x<-1.0) {
            return (1.5f * x + 5.0f) * x + 4.0f;
        } else if(x<0.0) {
            return (-4.5f * x - 5.0f) * x;
        } else if(x<1.0) {
            return (4.5f * x - 5.0f) * x;
        } else if(x<2.0) {
            return (-1.5f * x + 5.0f) * x - 4.0f;
        }
        return 0.0;
    }
    UNIAPI static T radius() { return static_cast<T>(2.f); }
};

template <typename T>
class ResamplerKernelHelper<ResamplingKernelType::BernsteinQuintic, T> {
// Bernstein, Gary M. and Gruen, Daniel, "Resampling images in Fourier domain"
public:
    UNIAPI static T value(T x) {
        x = std::abs(x);
        // The Compiler should be able to optimize the comparison sequence
        if (x <=1.0f) {
            return (((-55.0f*x + 138.0f)*x - 95.0f)*x*x*x + 12.0f)/12.0f;
        } else if (x <=2.0f) {
            return (((((55.0f*x - 414.0f)*x + 1205.0f)*x - 1680.0f)*x + 1110)*x - 276.0f)/24.0f;
        } else if (x <= 3.0f) {
            return (((((-11.0f*x + 138.0f)*x - 685.0f)*x + 1680.0f)*x - 2034.0f)*x + 972.0f)/24.0f;
        } else {
            return 0.0f;
        }
    }

    UNIAPI static T derivative(T x) {
        bool neg = std::signbit(x);
        x = std::abs(x);
        T ret;
        if (x <=1.0f) {
            ret = ((-275.0f*x + 552.0f)*x - 285.0f)*x*x/12.0f;
        } else if (x <=2.0f) {
            ret = ((((275.0f*x - 1656.0f)*x + 3615.0f)*x - 3360.0f)*x + 1110)/24.0f;
        } else if (x <= 3.0f) {
            return ((((-55.0f*x + 552.0f)*x - 2055.0f)*x +3360.0f)*x - 2034.0f)/24.0f;
        } else {
            ret = 0.0f;
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
