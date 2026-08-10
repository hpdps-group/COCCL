add_library(coccl_zfp_plugin SHARED zfp_plugin.cc)

target_compile_features(coccl_zfp_plugin PRIVATE cxx_std_17)
target_include_directories(coccl_zfp_plugin PRIVATE
  ${CUDA_INCLUDE_DIRS}
  ${COCCL_INCLUDE_DIRS}
)
# The build tree can be visible through multiple mount paths. Keep this as an
# explicit compiler option so a stale CMake implicit-include cache cannot drop it.
target_compile_options(coccl_zfp_plugin PRIVATE
  "-I${COCCL_NCCL_INCLUDE_DIR}"
)
target_link_libraries(coccl_zfp_plugin PRIVATE zfp ${CUDA_CUDART_LIBRARY})

set_target_properties(coccl_zfp_plugin PROPERTIES
  OUTPUT_NAME zfp
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  LINK_FLAGS "-Wl,--exclude-libs,ALL"
)

if(BUILD_COCCL)
  install(TARGETS coccl_zfp_plugin
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
