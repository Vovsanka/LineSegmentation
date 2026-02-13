# Allow user to specify GUROBI_ROOT or use GUROBI_HOME
set(_GUROBI_HINTS
    ${GUROBI_ROOT}
    ${GUROBI_ROOT_DIR}
    $ENV{GUROBI_HOME}
)

# Find include directory
find_path(GUROBI_INCLUDE_DIR
    NAMES gurobi_c.h
    HINTS ${_GUROBI_HINTS}
    PATH_SUFFIXES include
)

# Find main Gurobi library (version-agnostic)
find_library(GUROBI_LIBRARY
    NAMES gurobi gurobi90 gurobi91 gurobi100 gurobi110 gurobi120 gurobi130
    HINTS ${_GUROBI_HINTS}
    PATH_SUFFIXES lib
)

# Find C++ wrapper library
find_library(GUROBI_CXX_LIBRARY
    NAMES gurobi_c++ gurobi_c++3 gurobi_c++4
    HINTS ${_GUROBI_HINTS}
    PATH_SUFFIXES lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GUROBI DEFAULT_MSG
    GUROBI_LIBRARY
    GUROBI_CXX_LIBRARY
    GUROBI_INCLUDE_DIR
)

if(GUROBI_FOUND)
    add_library(Gurobi::gurobi UNKNOWN IMPORTED)
    set_target_properties(Gurobi::gurobi PROPERTIES
        IMPORTED_LOCATION "${GUROBI_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GUROBI_INCLUDE_DIR}"
    )

    add_library(Gurobi::cxx UNKNOWN IMPORTED)
    set_target_properties(Gurobi::cxx PROPERTIES
        IMPORTED_LOCATION "${GUROBI_CXX_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GUROBI_INCLUDE_DIR}"
    )
endif()
