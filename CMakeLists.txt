cmake_minimum_required(VERSION 3.15)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_softdtree STABLE_ABI src/main.cpp)
install(TARGETS _softdtree DESTINATION ${SKBUILD_PROJECT_NAME})
install(FILES $<TARGET_FILE_DIR:_softdtree>/_softdtree.pyi $<TARGET_FILE_DIR:_softdtree>/py.typed DESTINATION ${SKBUILD_PROJECT_NAME})

nanobind_add_stub(
  _softdtree_stub
  MODULE _softdtree
  OUTPUT _softdtree.pyi
  MARKER_FILE py.typed
  PYTHON_PATH $<TARGET_FILE_DIR:_softdtree>
  DEPENDS _softdtree
)

find_package(Eigen3 CONFIG REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(_softdtree PRIVATE OpenMP::OpenMP_CXX)
endif()

include(CheckCXXCompilerFlag)

set(MARCH_OPT "native")
check_cxx_compiler_flag("-march=${MARCH_OPT}" BUILD_MARCH_NATIVE)
if(BUILD_MARCH_NATIVE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${MARCH_OPT}")
else()
  unset(MARCH_OPT)
endif()

if (NOT MARCH_OPT)
  set(MCPU_OPT "native")
  check_cxx_compiler_flag("-mcpu=${MARCH_OPT}" BUILD_MCPU_NATIVE)
  if(BUILD_MCPU_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=${MCPU_OPT}")
  else()
    unset(MCPU_OPT)
  endif()
endif()
