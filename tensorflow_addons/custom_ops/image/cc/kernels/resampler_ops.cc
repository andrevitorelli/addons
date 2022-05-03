// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#define EIGEN_USE_THREADS

#include "tensorflow_addons/custom_ops/image/cc/kernels/resampler_ops.h"
#include "tensorflow_addons/custom_ops/image/cc/kernels/sampling_functions.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/work_sharder.h"



namespace tensorflow {

namespace addons {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


namespace functor {



template <ResamplingKernelType kernel_functor_class, typename T>
struct Resampler2DFunctor<CPUDevice, kernel_functor_class, T> {
  void operator()(OpKernelContext* ctx, const CPUDevice& d,
                  const T* __restrict__ data, const T* __restrict__ warp,
                  T* __restrict__ output, const int batch_size,
                  const int data_height, const int data_width,
                  const int data_channels, const int num_sampling_points) {
    const int warp_batch_stride = num_sampling_points * 2;
    const int data_batch_stride = data_height * data_width * data_channels;
    const int output_batch_stride = num_sampling_points * data_channels;
    const T zero = static_cast<T>(0.0);
    //const T one = static_cast<T>(1.0);

    using kernel = ResamplerKernelHelper<kernel_functor_class, T>;
    // Creating the interpolation kernel
    //auto kernel = ResamplerKernelHelper<kernel_functor_class>::createKernelFunction();

    auto resample_batches = [&](const int start, const int limit) {
      for (int batch_id = start; batch_id < limit; ++batch_id) {
        // Utility lambda to access data point and set output values.
        // The functions take care of performing the relevant pointer
        // arithmetics abstracting away the low level details in the
        // main loop over samples. Note that data is stored in NHWC format.
        auto set_output = [&](const int sample_id, const int channel,
                              const T value) {
          output[batch_id * output_batch_stride + sample_id * data_channels +
                 channel] = value;
        };

        auto get_data_point = [&](const int x, const int y, const int chan) {
          const bool point_is_in_range =
              (x >= 0 && y >= 0 && x <= data_width - 1 && y <= data_height - 1);
          return point_is_in_range
                     ? data[batch_id * data_batch_stride +
                            data_channels * (y * data_width + x) + chan]
                     : zero;
        };

        for (int sample_id = 0; sample_id < num_sampling_points; ++sample_id) {
          const T x = warp[batch_id * warp_batch_stride + sample_id * 2];
          const T y = warp[batch_id * warp_batch_stride + sample_id * 2 + 1];
          // The interpolation function:
          // a) implicitly pads the input data with 0s (hence the unusual checks
          // with {x,y} > -1)
          // b) returns 0 when sampling outside the (padded) image.
          // The effect is that the sampled signal smoothly goes to 0 outside
          // the original input domain, rather than presenting a jump
          // discontinuity at the image boundaries.
          if (x > static_cast<T>(-1.0) && y > static_cast<T>(-1.0) &&
              x < static_cast<T>(data_width) &&
              y < static_cast<T>(data_height)) {

            // Precompute floor (f) and ceil (c) values for x and y.
            const int fx = std::floor(x);
            const int fy = std::floor(y);

            const int span_size =
              static_cast<int>(std::ceil(kernel::radius()));
            for (int chan = 0; chan < data_channels; ++chan) {
              T res = zero;

              for(int inx=-span_size; inx <= span_size; inx++){
                for(int iny=-span_size; iny <= span_size; iny++){        
                  const int cx = fx + inx;
                  const int cy = fy + iny;
                  const T dx = static_cast<T>(cx) - x;
                  const T dy = static_cast<T>(cy) - y;
                  res += get_data_point(cx, cy, chan) * static_cast<T>(kernel::value(dx) * kernel::value(dy));
                }
              }
              set_output(sample_id, chan, res);
            }

          } else {
            for (int chan = 0; chan < data_channels; ++chan) {
              set_output(sample_id, chan, zero);
            }
          }
        }
      }
    };
    // Rough estimate of work for each batch entry.
    // From third_party/tensorflow/core/util/work_sharder.cc we gather that an
    // estimate of the cost of each work unit is needed to correctly shard the
    // workload. thread_pool->ParallelFor assumes each cost unit is 1ns, minimum
    // cost per shard
    // being 10us.
    const int64 cost =
        static_cast<int64>(num_sampling_points) * data_channels * 1000;
    auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    thread_pool->ParallelFor(batch_size, cost, resample_batches);
  }
};

}  // namespace functor


#include "register_kernel_factory.h"

#define REGISTER(TYPE)                                                       \
  REGISTER_KERNEL_FACTORY(                                                   \
      Name("Addons>Resampler").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ResamplerOpFactory<CPUDevice, TYPE>);

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
#undef REGISTER

namespace functor {

template <ResamplingKernelType kernel_functor_class, typename T>
struct ResamplerGrad2DFunctor<CPUDevice, kernel_functor_class, T> {
  void operator()(OpKernelContext* ctx, const CPUDevice& d,
                  const T* __restrict__ data, const T* __restrict__ warp,
                  const T* __restrict__ grad_output, T* __restrict__ grad_data,
                  T* __restrict__ grad_warp, const int batch_size,
                  const int data_height, const int data_width,
                  const int data_channels, const int num_sampling_points) {
    // Set gradients to 0, because the kernel incrementally updates the
    // tensor entries by adding partial contributions.
    const int resampler_output_size =
        batch_size * num_sampling_points * data_channels;
    const int grad_warp_size = resampler_output_size / data_channels * 2;
    const int grad_data_size =
        data_height * data_width * data_channels * batch_size;
    memset(static_cast<void*>(grad_data), 0, sizeof(T) * grad_data_size);
    memset(static_cast<void*>(grad_warp), 0, sizeof(T) * grad_warp_size);

    const auto&& data_batch_stride = data_height * data_width * data_channels;
    const auto&& warp_batch_stride = num_sampling_points * 2;
    const int output_batch_stride = num_sampling_points * data_channels;
    const T zero = static_cast<T>(0.0);
    //const T one = static_cast<T>(1.0);
    
    using kernel = ResamplerKernelHelper<kernel_functor_class, T>;
    // Creating the interpolation kernel and its 1st derivative
    //auto kernel = ResamplerKernelHelper<kernel_functor_class>::createKernelFunction();
    //auto kernelderivative = ResamplerKernelHelper<kernel_functor_class>::createKernelDerivativeFunction();
    
    auto update_grads_for_batches = [&](const int start, const int limit) {
      for (int batch_id = start; batch_id < limit; ++batch_id) {
        // Utility lambdas to access data and update gradient tensors.
        // The functions take care of performing the relevant pointer
        // arithmetics abstracting away the low level details in the
        // main loop over samples. Note that data is stored in NHWC format.
        auto get_data_point = [&](const int x, const int y, const int chan) {
          const bool point_is_in_range =
              (x >= 0 && y >= 0 && x <= data_width - 1 && y <= data_height - 1);
          return point_is_in_range
                     ? data[batch_id * data_batch_stride +
                            data_channels * (y * data_width + x) + chan]
                     : zero;
        };

        auto update_grad_data = [&](const int x, const int y, const int chan,
                                    const T value) {
          const bool point_is_in_range =
              (x >= 0 && y >= 0 && x <= data_width - 1 && y <= data_height - 1);
          if (point_is_in_range) {
            grad_data[batch_id * data_batch_stride +
                      data_channels * (y * data_width + x) + chan] += value;
          }
        };

        auto update_grad_warp = [&](const int sample_id, const int channel,
                                    const T value) {
          grad_warp[batch_id * warp_batch_stride + sample_id * 2 + channel] +=
              value;
        };

        for (int sample_id = 0; sample_id < num_sampling_points; ++sample_id) {
          const T x = warp[batch_id * warp_batch_stride + sample_id * 2];
          const T y = warp[batch_id * warp_batch_stride + sample_id * 2 + 1];
          // The interpolation function whose gradient this function implements:
          // a) implicitly pads the input data with 0s (hence the unusual checks
          // with {x,y} > -1)
          // b) returns 0 when sampling outside the (padded) image.
          // The effect is that the sampled signal smoothly goes to 0 outside
          // the original input domain, rather than presenting a jump
          // discontinuity at the image boundaries.
          if (x > static_cast<T>(-1.0) && y > static_cast<T>(-1.0) &&
              x < static_cast<T>(data_width) &&
              y < static_cast<T>(data_height)) {
            // Precompute floor (f) and ceil (c) values for x and y.
            const int fx = std::floor(x);
            const int fy = std::floor(y);
          
            for (int chan = 0; chan < data_channels; ++chan) {
            
              const T grad_output_value =
                  grad_output[batch_id * output_batch_stride +
                              sample_id * data_channels + chan];
              T ddx = zero; // Accumulator for warp derivative (x component)
              T ddy = zero; // Accumulator for warp derivative (y component)
              const int span_size = static_cast<int>(std::ceil(kernel::radius()));
              for(int inx=-span_size; inx <= span_size; inx++){
                for(int iny=-span_size; iny <= span_size; iny++){        
                  const int cx = fx + inx;
                  const int cy = fy + iny;
                  const T dx = static_cast<T>(cx) - x;
                  const T dy = static_cast<T>(cy) - y;
                  auto val = get_data_point(cx, cy, chan);
                  
                  auto kernel_x = kernel::value(dx);
                  auto kernel_y = kernel::value(dy);
                  
                  ddx -= val * static_cast<T>(kernel::derivative(dx) * kernel_y);
                  ddy -= val * static_cast<T>(kernel::derivative(dy) * kernel_x);
                  
                  // Update partial gradients wrt sampled data
                  update_grad_data(cx, cy, chan, grad_output_value*static_cast<T>(kernel_x*kernel_y));
                }
              }
              // Update partial gradients wrt relevant warp field entries
              update_grad_warp(sample_id, 0, grad_output_value*ddx);
              update_grad_warp(sample_id, 1, grad_output_value*ddy);
              
            }
          }
        }
      }
    };
    // Rough estimate of work for each batch entry.
    // From third_party/tensorflow/core/util/work_sharder.cc we gather that an
    // estimate of the cost of each work unit is needed to correctly shard the
    // workload. thread_pool->ParallelFor assumes each cost unit is 1ns, minimum
    // cost per shard
    // being 10us.
    // TODO(fviola): Check out if there is a better way of doing this.
    auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    const int64 cost =
        static_cast<int64>(num_sampling_points) * data_channels * 1000;
    thread_pool->ParallelFor(batch_size, cost, update_grads_for_batches);
  }
};

}  // namespace functor


#define REGISTER(TYPE)                                    \
  REGISTER_KERNEL_FACTORY(Name("Addons>ResamplerGrad")    \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<TYPE>("T"), \
                          ResamplerGradOpFactory<CPUDevice, TYPE>);

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
#undef REGISTER

}  // end namespace addons
}  // namespace tensorflow
