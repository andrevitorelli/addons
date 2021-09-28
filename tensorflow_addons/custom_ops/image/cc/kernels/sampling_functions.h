#ifndef SAMPLING_FUNCTIONS_H_
#define SAMPLING_FUNCTIONS_H_

#ifdef __CUDACC__
#define UNIAPI  __device__
#else
#define UNIAPI
#endif


template <typename kernel_class, typename T>
class ResamplerKernelHelper {
};

template <typename T>
class ResamplerKernelHelper<tensorflow::functor::TriangleKernelFunc, T> {
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
    /*
    // 1st derivative of TriangleKernelFunc
    struct TriangleKernelDerivativeFunc {
        // Strictly speaking, Triangle Kernel is non-differentiable on -1, 0 and 1
        float operator()(float x) const {
            if(x<-1.0)
                return 0;
            if(x<0.0)
                return 1.0;
            if(x<1.0)
                return -1.0;
            return 0.0;
        }
        float Radius() const { return 1.f; }
    };

    static inline TriangleKernelDerivativeFunc createKernelDerivativeFunction() {
        return TriangleKernelDerivativeFunc();
    }*/
};

template <typename T>
class ResamplerKernelHelper< tensorflow::functor::KeysCubicKernelFunc, T> {
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
    /*
    typedef  tensorflow::functor::KeysCubicKernelFunc kernelFunc;
    static inline kernelFunc createKernelFunction() {
        return tensorflow::functor::CreateKeysCubicKernel();
    }

    struct KeysCubicKernelDerivativeFunc {
        // http://ieeexplore.ieee.org/document/1163711/
        // R. G. Keys. Cubic convolution interpolation for digital image
        // processing. IEEE Transactions on Acoustics, Speech, and Signal
        // Processing, 29(6):1153–1160, 1981.
        float operator()(float x) const {
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
        float Radius() const { return 2.f; }
    };

    static inline KeysCubicKernelDerivativeFunc createKernelDerivativeFunction() {
        return KeysCubicKernelDerivativeFunc();
    }*/
};

#endif  // SAMPLING_FUNCTIONS_H_
