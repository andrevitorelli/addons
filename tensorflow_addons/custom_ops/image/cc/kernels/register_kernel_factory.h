#ifndef TENSORFLOW_ADDONS_IMAGE_KERNELS_REGISTER_KERNEL_FACTORY_H_
#define TENSORFLOW_ADDONS_IMAGE_KERNELS_REGISTER_KERNEL_FACTORY_H_

// Now here's the deal: Tensorflow's op registration system uses a farily complex cpp api 
//  that involves singleton initialization (see the OpKernelRegistrar class defined in op_kernel.h).
// This api is wrapped in REGISTER_KERNEL_BUILDER macros. Now, although the original api fully supports the factory pattern,
//  this is not exposed by macros.
// We have two choices here: Either declare the underlying singleton initializations directly, wich will probably result in less
//  readable code, or add new macros hoping to someday get them included in tensorflow. We're doing the later here.

#ifndef REGISTER_KERNEL_FACTORY // So when (if) this gets into tensorflow, it won't break in our face
// The following is (minimally) adapted from tensorflow's own op_kernel.h
#define REGISTER_KERNEL_FACTORY_IMPL_3(ctr, op_name, kernel_builder_expr,   \
                                       is_system_kernel, ...)               \
  static ::tensorflow::InitOnStartupMarker const register_kernel_##ctr      \
      TF_ATTRIBUTE_UNUSED =                                                 \
          TF_INIT_ON_STARTUP_IF(is_system_kernel ||                         \
                                (SHOULD_REGISTER_OP_KERNEL(#__VA_ARGS__) && \
                                 SHOULD_REGISTER_OP(op_name)))              \
          << ([](::tensorflow::KernelDef const* kernel_def) {               \
               ::tensorflow::kernel_factory::OpKernelRegistrar registrar(   \
                   kernel_def, #__VA_ARGS__,                                \
                   absl::make_unique<__VA_ARGS__>());                       \
               (void)registrar;                                             \
               return ::tensorflow::InitOnStartupMarker{};                  \
             })(kernel_builder_expr.Build());

#define REGISTER_KERNEL_FACTORY_IMPL_2(op_name, kernel_builder_expr, \
                                       is_system_kernel, ...)        \
  TF_NEW_ID_FOR_INIT(REGISTER_KERNEL_FACTORY_IMPL_3, op_name,        \
                     kernel_builder_expr, is_system_kernel, __VA_ARGS__)

#define REGISTER_KERNEL_FACTORY_IMPL(kernel_builder, is_system_kernel, ...) \
  TF_EXTRACT_KERNEL_NAME(REGISTER_KERNEL_FACTORY_IMPL_2, kernel_builder,    \
                         is_system_kernel, __VA_ARGS__)

#define REGISTER_KERNEL_FACTORY(kernel_builder, ...) \
  TF_ATTRIBUTE_ANNOTATE("tf:kernel")                 \
  REGISTER_KERNEL_FACTORY_IMPL(kernel_builder, false, __VA_ARGS__)
  
#define REGISTER_SYSTEM_FACTORY_BUILDER(kernel_builder, ...) \
  TF_ATTRIBUTE_ANNOTATE("tf:kernel")                        \
  TF_ATTRIBUTE_ANNOTATE("tf:kernel:system")                 \
  REGISTER_KERNEL_FACTORY_IMPL(kernel_builder, true, __VA_ARGS__)
#endif 

#endif  // TENSORFLOW_ADDONS_IMAGE_KERNELS_REGISTER_KERNEL_FACTORY_H_
