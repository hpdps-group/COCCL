# if(ZFP_WITH_CUDA)
#   enable_language(CUDA)
# endif()
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 对 CUDA 代码启用 C++11
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(zfp_compressor SHARED zfp.c)
set_source_files_properties(zfp.c PROPERTIES LANGUAGE CXX)

# protect against LNK1114: cannot overwrite the original file 'lib/Release/zfp.lib'; error code 32;
# rationale: linker can't handle the case of an executable file having the same name as a library file
# if(NOT MSVC)
#   set_property(TARGET zfpcmd PROPERTY OUTPUT_NAME zfp)
# endif()
target_link_libraries(zfp_compressor PRIVATE zfp)

if(HAVE_LIBM_MATH)
  target_link_libraries(zfp_compressor PRIVATE m)
endif()


if(ZFP_WITH_CUDA)
  # target_compile_definitions(zfpcmd PRIVATE BF16_AVAILABLE) 
  set(COCCL_INCLUDE_DIRS ../../../../include CACHE STRING "COCCL_INCLUDE_DIRS")
  set(NCCL_INCLUDE_DIRS ../../../../../build/include CACHE STRING "NCCL_INCLUDE_DIRS")


  target_include_directories(zfp_compressor PRIVATE ${CUDA_INCLUDE_DIRS} ${COCCL_INCLUDE_DIRS} ${NCCL_INCLUDE_DIRS})
  target_link_libraries(zfp_compressor PRIVATE ${CUDA_CUDART_LIBRARY} stdc++)
endif()


if(BUILD_COCCL)
  install(TARGETS zfp_compressor EXPORT zfp_compressor-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# set_target_properties(zfp_compressor PROPERTIES
#     LIBRARY_OUTPUT_DIRECTORY "../../../../../build/obj/device/compress/libcompress"
# )

# set_property(TARGET zfp PROPERTY OUTPUT_NAME ${ZFP_LIBRARY_PREFIX}zfp)
message(LIBRARY_OUTPUT_DIRECTORY "LIBRARY_OUTPUT_DIRECTORY: ${LIBRARY_OUTPUT_DIRECTORY}")

# install(TARGETS zfp_compressor EXPORT zfp_compressor-targets
#   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
