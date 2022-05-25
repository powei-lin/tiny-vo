add_subdirectory(basalt-headers)

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(PROVIDE_UNINSTALL_TARGET OFF CACHE BOOL "" FORCE)
add_subdirectory(ceres-solver)

set(BUILD_PANGOLIN_PYTHON OFF CACHE BOOL "" FORCE)
add_subdirectory(Pangolin)