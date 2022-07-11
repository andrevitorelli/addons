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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>

#include <cmath>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow_addons/custom_ops/image/cc/kernels/resampler_ops.h"
#include "tensorflow_addons/custom_ops/image/cc/kernels/sampling_functions.h"

namespace tensorflow {
namespace addons {

using GPUDevice = Eigen::GpuDevice;

namespace {

#define GET_DATA_POINT(x, y)                                                 \
  data[batch_id * data_batch_stride + data_channels * (y * data_width + x) + \
       chan]

template <ResamplingKernelType kernel_functor_class, typename T>
__global__ void Resampler2DKernel(const T* __restrict__ data,
                                  const T* __restrict__ warp,
                                  T* __restrict__ output, const int batch_size,
                                  const int data_height, const int data_width,
                                  const int data_channels,
                                  const int num_sampling_points) {
  const int output_data_size = batch_size * num_sampling_points * data_channels;
  // Creating the interpolation kernel
  //auto kernel = ResamplerKernelHelper<kernel_functor_class>::createKernelFunction();
  using kernel = tensorflow::addons::functor::ResamplerKernelHelper<kernel_functor_class, float>;
  GPU_1D_KERNEL_LOOP(index, output_data_size) {
    const int out_index = index;

    // Get (idxSample, channel, point) from the index.
    // Use this formula
    //   index = batch_id * num_sampling_points * num_chans +
    //           sample_id * num_chans + chan_id,
    // with sample_id = [0, ... ,num_sampling_points)
    const int data_batch_stride = data_height * data_width * data_channels;
    const int warp_batch_stride = num_sampling_points * 2;
    const int output_batch_stride = num_sampling_points * data_channels;

    const int batch_id = index / output_batch_stride;
    const int index_in_batch = index % output_batch_stride;
    const int chan = index_in_batch % data_channels;
    const int sample_id = index_in_batch / data_channels;

    // Get coords of 2D point where data will be resampled
    const T x = warp[batch_id * warp_batch_stride + sample_id * 2];
    const T y = warp[batch_id * warp_batch_stride + sample_id * 2 + 1];
    const T zero = static_cast<T>(0.0);
    //const T one = static_cast<T>(1.0);
    // The interpolation function:
    // a) implicitly pads the input data with 0s (hence the unusual checks
    // with {x,y} > -1)
    // b) returns 0 when sampling outside the (padded) image.
    // The effect is that the sampled signal smoothly goes to 0 outside
    // the original input domain, rather than presenting a jump
    // discontinuity at the image boundaries.
    if (x > static_cast<T>(-1.0) && y > static_cast<T>(-1.0) &&
        x < static_cast<T>(data_width) && y < static_cast<T>(data_height)) {
      // Precompute floor (f) and ceil (c) values for x and y.
      const int fx = std::floor(static_cast<T>(x));
      const int fy = std::floor(static_cast<T>(y));
      const int span_size = static_cast<int>(std::ceil(kernel::radius()));
      T res = zero;
      for(int inx=-span_size; inx <= span_size; inx++){
        for(int iny=-span_size; iny <= span_size; iny++){
          const int sx = fx + inx; // Sampled coordinate
          const int sy = fy + iny;
          const T dx = static_cast<T>(sx) - x;
          const T dy = static_cast<T>(sy) - y;
          if(sx>=0 && sy>=0 && sx<data_width && sy < data_height)
            res += GET_DATA_POINT(sx, sy) * static_cast<T>(kernel::value(dx) * kernel::value(dy));
        }
      }
      output[out_index] = res;
       //set_output(sample_id, chan, res);
       //output[out_index] = img_fxfy + img_cxcy + img_fxcy + img_cxfy;
    } else {
      output[out_index] = zero;
    }
  }
}

}  // namespace

namespace functor {

template <ResamplingKernelType kernel_functor_class, typename T>
struct Resampler2DFunctor<GPUDevice, kernel_functor_class, T> {
  void operator()(OpKernelContext* ctx, const GPUDevice& d,
                  const T* __restrict__ data, const T* __restrict__ warp,
                  T* __restrict__ output, const int batch_size,
                  const int data_height, const int data_width,
                  const int data_channels, const int num_sampling_points) {
    const int output_data_size =
        batch_size * num_sampling_points * data_channels;
    GpuLaunchConfig config = GetGpuLaunchConfig(output_data_size, d);
    TF_CHECK_OK(GpuLaunchKernel(
        Resampler2DKernel<kernel_functor_class, T>, config.block_count, config.thread_per_block, 0,
        d.stream(), data, warp, output, batch_size, data_height, data_width,
        data_channels, num_sampling_points));
  }
};

}  // namespace functor

namespace {

#define UPDATE_GRAD_DATA_POINT(x, y, v)                                   \
  GpuAtomicAdd(grad_data + (batch_id * data_batch_stride +                \
                            data_channels * (y * data_width + x) + chan), \
               v)

template <ResamplingKernelType kernel_functor_class, typename T>
__global__ void ResamplerGrad2DKernel(
    const T* __restrict__ data, const T* __restrict__ warp,
    const T* __restrict__ grad_output, T* __restrict__ grad_data,
    T* __restrict__ grad_warp, const int batch_size, const int data_height,
    const int data_width, const int data_channels,
    const int num_sampling_points) {
  const int resampler_output_size =
      batch_size * num_sampling_points * data_channels;
  using kernel = tensorflow::addons::functor::ResamplerKernelHelper<kernel_functor_class, T>;
  GPU_1D_KERNEL_LOOP(index, resampler_output_size) {
    const int out_index = index;

    // Get (idxSample, channel, point) from the index.
    // Use this formula
    //   index = batch_id * num_sampling_points * num_chans +
    //           sample_id * num_chans + chan_id,
    // with sample_id = [0, ... ,num_sampling_points)
    const int data_batch_stride = data_height * data_width * data_channels;
    const int warp_batch_stride = num_sampling_points * 2;
    const int output_batch_stride = num_sampling_points * data_channels;

    const int batch_id = index / output_batch_stride;
    const int index_in_batch = index % output_batch_stride;
    const int chan = index_in_batch % data_channels;
    const int sample_id = index_in_batch / data_channels;

    // Get coords of 2D point where data will be resampled
    const int warp_id_x = batch_id * warp_batch_stride + sample_id * 2;
    const int warp_id_y = warp_id_x + 1;
    const T x = warp[warp_id_x];
    const T y = warp[warp_id_y];
    const T zero = static_cast<T>(0.0);
    //const T one = static_cast<T>(1.0);

    // Get grad output
    const T grad_output_value = grad_output[out_index];
    // The interpolation function whose gradient this kernel implements:
    // a) implicitly pads the input data with 0s (hence the unusual checks
    // with {x,y} > -1)
    // b) returns 0 when sampling outside the (padded) image.
    // The effect is that the sampled signal smoothly goes to 0 outside
    // the original input domain, rather than presenting a jump
    // discontinuity at the image boundaries.
    if (x > static_cast<T>(-1.0) && y > static_cast<T>(-1.0) &&
        x < static_cast<T>(data_width) && y < static_cast<T>(data_height)) {
      // Precompute floor (f) and ceil (c) values for x and y.
      const int fx = std::floor(x);
      const int fy = std::floor(y);

      T ddx = zero; // Accumulator for warp derivative (x component)
      T ddy = zero; // Accumulator for warp derivative (y component)

      const int span_size = static_cast<int>(std::ceil(kernel::radius()));
      for(int inx=-span_size; inx <= span_size; inx++){
        for(int iny=-span_size; iny <= span_size; iny++){
                const int sx = fx + inx;
                const int sy = fy + iny;
                const T dx = static_cast<T>(sx) - x;
                const T dy = static_cast<T>(sy) - y;
                if(sx >= 0 && sy >= 0 && sx < data_width && sy < data_height) {
                    auto val = GET_DATA_POINT(sx, sy);

                    auto kernel_x = kernel::value(dx);
                    auto kernel_y = kernel::value(dy);
                    ddx -= val * static_cast<T>(kernel::derivative(dx) * kernel_y);
                    ddy -= val * static_cast<T>(kernel::derivative(dy) * kernel_x);

                    // Update partial gradients wrt sampled data
                    UPDATE_GRAD_DATA_POINT(sx, sy, static_cast<T>(grad_output_value * kernel_x * kernel_y));
                }
            }
      }
      // Update partial gradients wrt relevant warp field entries
      *(grad_warp + warp_id_x) +=  static_cast<T>(grad_output_value * ddx);
      *(grad_warp + warp_id_y) +=  static_cast<T>(grad_output_value * ddy);
    }
  }
}

#undef GET_DATA_POINT
#undef UPDATE_GRAD_DATA_POINT

}  // namespace

namespace functor {

template <ResamplingKernelType kernel_functor_class, typename T>
struct ResamplerGrad2DFunctor<GPUDevice, kernel_functor_class, T> {
  void operator()(OpKernelContext* ctx, const GPUDevice& d,
                  const T* __restrict__ data, const T* __restrict__ warp,
                  const T* __restrict__ grad_output, T* __restrict__ grad_data,
                  T* __restrict__ grad_warp, const int batch_size,
                  const int data_height, const int data_width,
                  const int data_channels, const int num_sampling_points) {
    // Set gradients to 0, because the kernel incrementally updates the
    // tensor entries by adding partial contributions.
    const int grad_warp_size = batch_size * num_sampling_points * 2;
    const int grad_data_size =
        batch_size * data_height * data_width * data_channels;

    GpuLaunchConfig config = GetGpuLaunchConfig(grad_warp_size, d);
    TF_CHECK_OK(GpuLaunchKernel(SetZero<T>, config.block_count,
                                config.thread_per_block, 0, d.stream(),
                                grad_warp_size, grad_warp));

    config = GetGpuLaunchConfig(grad_data_size, d);
    TF_CHECK_OK(GpuLaunchKernel(SetZero<T>, config.block_count,
                                config.thread_per_block, 0, d.stream(),
                                grad_data_size, grad_data));

    const int resampler_output_size =
        batch_size * num_sampling_points * data_channels;
    config = GetGpuLaunchConfig(resampler_output_size, d);
    TF_CHECK_OK(GpuLaunchKernel(ResamplerGrad2DKernel<kernel_functor_class, T>, config.block_count,
                                config.thread_per_block, 0, d.stream(), data,
                                warp, grad_output, grad_data, grad_warp,
                                batch_size, data_height, data_width,
                                data_channels, num_sampling_points));
  }
};
}  // namespace functor

#include "register_kernel_factory.h"

#define REGISTER(TYPE)                                                       \
  REGISTER_KERNEL_FACTORY(                                                   \
      Name("Addons>Resampler").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      ResamplerOpFactory<GPUDevice, TYPE>)

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
#undef REGISTER

#define REGISTER(TYPE)                                    \
  REGISTER_KERNEL_FACTORY(Name("Addons>ResamplerGrad")    \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<TYPE>("T"), \
                          ResamplerGradOpFactory<GPUDevice, TYPE>)

TF_CALL_half(REGISTER);
TF_CALL_double(REGISTER);
TF_CALL_float(REGISTER);
#undef REGISTER


}  // namespace addons
}  // namespace tensorflow


#endif  // GOOGLE_CUDA
