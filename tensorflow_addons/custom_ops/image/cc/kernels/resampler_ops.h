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

#ifndef TENSORFLOW_ADDONS_IMAGE_KERNELS_RESAMPLER_OPS_H_
#define TENSORFLOW_ADDONS_IMAGE_KERNELS_RESAMPLER_OPS_H_

#if PLATFORM_WINDOWS
#define __restrict__ __restrict
#endif

#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/image/sampling_kernels.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"


namespace tensorflow {
namespace addons {

enum class ResamplingKernelType {
      Triangle=0, KeysCubic, BernsteinQuintic, Unknown
};


// FIXME: Move it to another file?
inline ResamplingKernelType Str2SamplingKernelType(const StringPiece str) {
  const string lower_case = absl::AsciiStrToLower(str);
  if (lower_case == "triangle") return ResamplingKernelType::Triangle;
  if (lower_case == "keyscubic") return ResamplingKernelType::KeysCubic;
  if (lower_case == "bernsteinquintic") return ResamplingKernelType::BernsteinQuintic;
  return ResamplingKernelType::Unknown;
}

namespace functor {

// Helper functor for the Resampler Op in 2D
template <typename Device, ResamplingKernelType kernel_functor_class, typename T>
struct Resampler2DFunctor {
  void operator()(OpKernelContext* ctx, const Device& d,
                  const T* __restrict__ data, const T* __restrict__ warp,
                  T* __restrict__ output, const int batch_size,
                  const int data_height, const int data_width,
                  const int data_channels, const int num_sampling_points);
};

// Helper functor for the Resampler Gradient Op in 2D
template <typename Device, ResamplingKernelType kernel_functor_class, typename T>
struct ResamplerGrad2DFunctor {
  void operator()(OpKernelContext* ctx, const Device& d,
                  const T* __restrict__ data, const T* __restrict__ warp,
                  const T* __restrict__ grad_output, T* __restrict__ grad_data,
                  T* __restrict__ grad_warp, const int batch_size,
                  const int data_height, const int data_width,
                  const int data_channels, const int num_sampling_points);
};

}  // namespace functor

template <typename Device, ResamplingKernelType kernel_functor_class, typename T>
class ResamplerOp : public OpKernel {
 public:
  explicit ResamplerOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& data = ctx->input(0);
    const Tensor& warp = ctx->input(1);

    const TensorShape& data_shape = data.shape();
    OP_REQUIRES(ctx, data_shape.dims() == 4,
                errors::Unimplemented(
                    "Only bilinear interpolation is currently supported. The "
                    "input data shape must be [batch_size, data_height, "
                    "data_width, data_channels], but is: ",
                    data_shape.DebugString()));
    const TensorShape& warp_shape = warp.shape();
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsMatrixOrHigher(warp_shape),
        errors::InvalidArgument("warp should be at least a matrix, got shape ",
                                "warp should be at least a matrix, got shape ",
                                warp_shape.DebugString()));
    OP_REQUIRES(ctx, warp_shape.dim_size(warp_shape.dims() - 1) == 2,
                errors::Unimplemented(
                    "Only bilinear interpolation is supported, warping "
                    "coordinates must be 2D; warp shape last entry should be "
                    "2, but shape vector is: ",
                    warp_shape.DebugString()));
    OP_REQUIRES(ctx, data_shape.dim_size(0) == warp_shape.dim_size(0),
                errors::InvalidArgument(
                    "Batch size of data and warp tensor must be the same, but "
                    "input shapes are: ",
                    data_shape.DebugString(), ", ", warp_shape.DebugString()));
    const int batch_size = data_shape.dim_size(0);
    const int data_height = data_shape.dim_size(1);
    const int data_width = data_shape.dim_size(2);
    const int data_channels = data_shape.dim_size(3);
    TensorShape output_shape = warp.shape();
    output_shape.set_dim(output_shape.dims() - 1, data_channels);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    // Execute kernel only for nonempty output; otherwise Eigen crashes on GPU.
    if (data.NumElements() > 0 && warp.NumElements() > 0) {
      const int num_sampling_points = warp.NumElements() / batch_size / 2;
      functor::Resampler2DFunctor<Device, kernel_functor_class, T>()(
          ctx, ctx->eigen_device<Device>(), data.flat<T>().data(),
          warp.flat<T>().data(), output->flat<T>().data(), batch_size,
          data_height, data_width, data_channels, num_sampling_points);
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ResamplerOp);
};

template <typename Device, typename T>
class ResamplerOpFactory: public ::tensorflow::kernel_factory::OpKernelFactory {
    OpKernel* Create(OpKernelConstruction* context) override {
        string kernel_type_str;
        ::tensorflow::Status s(context->GetAttr("kernel_type", &kernel_type_str));
        if(!TF_PREDICT_TRUE(s.ok())) {
            context->CtxFailureWithWarning(__FILE__, __LINE__, s);
            return nullptr;
        }
        ResamplingKernelType kernel_type_ = Str2SamplingKernelType(kernel_type_str);

        OpKernel *kernel = nullptr;

        switch(kernel_type_) {
            case ResamplingKernelType::Triangle:
                kernel =  new ResamplerOp<Device, ResamplingKernelType::Triangle, T>(context);
                break;

            case ResamplingKernelType::KeysCubic:
                kernel = new ResamplerOp<Device, ResamplingKernelType::KeysCubic, T>(context);
                break;

            case ResamplingKernelType::BernsteinQuintic:
                kernel = new ResamplerOp<Device, ResamplingKernelType::BernsteinQuintic, T>(context);
                break;

            case ResamplingKernelType::Unknown:
                context->CtxFailure(__FILE__, __LINE__,
                    errors::InvalidArgument("Unrecognized kernel type: " + kernel_type_str));
                break;

            default:
                context->CtxFailure(__FILE__, __LINE__,
                    errors::InvalidArgument("Unsupported kernel type: " + kernel_type_str));
                break;
        }

        return kernel;
    }
};

template <typename Device, ResamplingKernelType kernel_functor_class, typename T>
class ResamplerGradOp : public OpKernel {
 public:
  explicit ResamplerGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& data = ctx->input(0);
    const Tensor& warp = ctx->input(1);
    const Tensor& grad_output = ctx->input(2);

    const TensorShape& data_shape = data.shape();
    OP_REQUIRES(ctx, data_shape.dims() == 4,
                errors::Unimplemented(
                    "Only bilinear interpolation is supported, the input data "
                    "tensor must be a batch of 2d data; data shape should have "
                    "4 entries corresponding to [batch_size, data_height, "
                    "data_width, data_channels], but is: ",
                    data_shape.DebugString()));
    const int batch_size = data_shape.dim_size(0);
    const int data_height = data_shape.dim_size(1);
    const int data_width = data_shape.dim_size(2);
    const int data_channels = data_shape.dim_size(3);
    const TensorShape& warp_shape = warp.shape();
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsMatrixOrHigher(warp_shape),
        errors::InvalidArgument("warp should be at least a matrix, got shape ",
                                warp_shape.DebugString()));
    OP_REQUIRES(ctx, warp_shape.dim_size(warp_shape.dims() - 1) == 2,
                errors::Unimplemented(
                    "Only bilinear interpolation is supported, warping "
                    "coordinates must be 2D; warp shape last entry should be "
                    "2, but shape vector is: ",
                    warp_shape.DebugString()));
    const TensorShape& grad_output_shape = grad_output.shape();
    TensorShape resampler_output_shape = warp.shape();
    resampler_output_shape.set_dim(resampler_output_shape.dims() - 1,
                                   data_channels);
    OP_REQUIRES(ctx, grad_output_shape == resampler_output_shape,
                errors::InvalidArgument(
                    "grad_output shape is not consistent with data and warp "
                    "shapes; it should be ",
                    resampler_output_shape.DebugString(), " but is ",
                    grad_output_shape.DebugString()));
    Tensor* grad_data = nullptr;
    Tensor* grad_warp = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, data.shape(), &grad_data));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, warp.shape(), &grad_warp));
    // Execute kernel only for nonempty output; otherwise Eigen crashes on GPU.
    if (data.NumElements() > 0 && warp.NumElements() > 0) {
      const int num_sampling_points = warp.NumElements() / batch_size / 2;
      functor::ResamplerGrad2DFunctor<Device, kernel_functor_class, T>()(
          ctx, ctx->eigen_device<Device>(), data.flat<T>().data(),
          warp.flat<T>().data(), grad_output.flat<T>().data(),
          grad_data->flat<T>().data(), grad_warp->flat<T>().data(), batch_size,
          data_height, data_width, data_channels, num_sampling_points);
    }
  }
 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ResamplerGradOp);
};

template <typename Device, typename T>
class ResamplerGradOpFactory: public ::tensorflow::kernel_factory::OpKernelFactory {
    OpKernel* Create(OpKernelConstruction* context) override {
        string kernel_type_str;
        ::tensorflow::Status s(context->GetAttr("kernel_type", &kernel_type_str));
        if(!TF_PREDICT_TRUE(s.ok())) {
            context->CtxFailureWithWarning(__FILE__, __LINE__, s);
            return nullptr;
        }

        ResamplingKernelType kernel_type_ = Str2SamplingKernelType(kernel_type_str);

        OpKernel *kernel = nullptr;

        switch(kernel_type_) {
              case ResamplingKernelType::Triangle:
                kernel =  new ResamplerGradOp<Device, ResamplingKernelType::Triangle, T>(context);
                break;

            case ResamplingKernelType::KeysCubic:
                kernel = new ResamplerGradOp<Device, ResamplingKernelType::KeysCubic, T>(context);
                break;

            case ResamplingKernelType::BernsteinQuintic:
                kernel = new ResamplerGradOp<Device, ResamplingKernelType::BernsteinQuintic, T>(context);
                break;

            case ResamplingKernelType::Unknown:
                context->CtxFailure(__FILE__, __LINE__,
                    errors::InvalidArgument("Unrecognized kernel type: " + kernel_type_str));
                break;

            default:
                context->CtxFailure(__FILE__, __LINE__,
                    errors::InvalidArgument("Unsupported kernel type: " + kernel_type_str));
                break;
        }

        return kernel;
    }
};

}  // namespace addons
}  // namespace tensorflow

#endif  // TENSORFLOW_ADDONS_IMAGE_KERNELS_RESAMPLER_OPS_H_
