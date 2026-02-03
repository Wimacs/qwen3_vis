cmake_minimum_required(VERSION 3.16)

project(qwen3_c C)

option(QWEN3_ENABLE_OPENMP "Enable OpenMP parallelism" ON)
option(QWEN3_BUILD_GUI "Build raylib GUI application" ON)

# ----------------------------------------------------------------------------
# qwen3 library (shared inference code)
add_library(qwen3_lib STATIC
    qwen3.c
    win.c
)
target_include_directories(qwen3_lib PUBLIC .)

if(MSVC)
    target_compile_options(qwen3_lib PRIVATE /fp:fast /Ox)
    if(QWEN3_ENABLE_OPENMP)
        target_compile_options(qwen3_lib PRIVATE /openmp:llvm)
    endif()
else()
    target_compile_options(qwen3_lib PRIVATE -O3 -ffast-math)
    if(QWEN3_ENABLE_OPENMP)
        find_package(OpenMP QUIET)
        if(OpenMP_C_FOUND)
            target_link_libraries(qwen3_lib PUBLIC OpenMP::OpenMP_C)
        endif()
    endif()
endif()

# ----------------------------------------------------------------------------
# runq CLI executable
add_executable(runq
    runq.c
    win.c
)

target_include_directories(runq PRIVATE .)

if(MSVC)
    target_compile_options(runq PRIVATE /fp:fast /Ox)
    if(QWEN3_ENABLE_OPENMP)
        target_compile_options(runq PRIVATE /openmp:llvm)
    endif()
else()
    target_compile_options(runq PRIVATE -O3 -ffast-math)
    if(QWEN3_ENABLE_OPENMP)
        find_package(OpenMP QUIET)
        if(OpenMP_C_FOUND)
            target_link_libraries(runq PRIVATE OpenMP::OpenMP_C)
        endif()
    endif()
endif()

# ----------------------------------------------------------------------------
# raylib GUI executable
if(QWEN3_BUILD_GUI)
    # raylib configuration
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(PLATFORM "SDL" CACHE STRING "" FORCE)

    # SDL2 (submodule) configuration
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    set(SDL_STATIC ON CACHE BOOL "" FORCE)
    set(SDL_TEST OFF CACHE BOOL "" FORCE)
    add_subdirectory(3rd/SDL2 EXCLUDE_FROM_ALL)
    
    # Add raylib as subdirectory
    add_subdirectory(3rd/raylib)

    # Force SDL platform define on raylib target to ensure SDL backend is compiled
    target_compile_definitions(raylib PRIVATE PLATFORM_DESKTOP_SDL)
    target_compile_definitions(raylib PUBLIC PLATFORM_DESKTOP_SDL)
    
    # GUI executable
    add_executable(runq_gui
        runq_gui.c
    )
    
    target_link_libraries(runq_gui PRIVATE qwen3_lib raylib)
    target_include_directories(runq_gui PRIVATE . 3rd/raylib/src)
    
    if(MSVC)
        target_compile_options(runq_gui PRIVATE /fp:fast /Ox)
        if(QWEN3_ENABLE_OPENMP)
            target_compile_options(runq_gui PRIVATE /openmp:llvm)
        endif()
        # Link Windows libraries for raylib
        target_link_libraries(runq_gui PRIVATE winmm imm32)
    else()
        target_compile_options(runq_gui PRIVATE -O3 -ffast-math)
        if(QWEN3_ENABLE_OPENMP)
            find_package(OpenMP QUIET)
            if(OpenMP_C_FOUND)
                target_link_libraries(runq_gui PRIVATE OpenMP::OpenMP_C)
            endif()
        endif()
        # Link platform-specific libraries
        if(WIN32)
            target_link_libraries(runq_gui PRIVATE winmm imm32)
        elseif(APPLE)
            target_link_libraries(runq_gui PRIVATE "-framework IOKit" "-framework Cocoa" "-framework OpenGL")
        elseif(UNIX)
            target_link_libraries(runq_gui PRIVATE m pthread dl GL X11)
        endif()
    endif()
    
    # Copy model and tokenizer files to build directory for convenience
    if(EXISTS "${CMAKE_SOURCE_DIR}/Qwen3-0.6B.bin")
        add_custom_command(TARGET runq_gui POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/Qwen3-0.6B.bin"
            "$<TARGET_FILE_DIR:runq_gui>/Qwen3-0.6B.bin"
        )
    endif()
    if(EXISTS "${CMAKE_SOURCE_DIR}/Qwen3-0.6B.bin.tokenizer")
        add_custom_command(TARGET runq_gui POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/Qwen3-0.6B.bin.tokenizer"
            "$<TARGET_FILE_DIR:runq_gui>/Qwen3-0.6B.bin.tokenizer"
        )
    endif()
    # Copy prompt template files (used by chat_init)
    foreach(_tpl template template.with-system template.with-thinking template.with-system-and-thinking)
        if(EXISTS "${CMAKE_SOURCE_DIR}/Qwen3-0.6B.bin.${_tpl}")
            add_custom_command(TARGET runq_gui POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_SOURCE_DIR}/Qwen3-0.6B.bin.${_tpl}"
                "$<TARGET_FILE_DIR:runq_gui>/Qwen3-0.6B.bin.${_tpl}"
            )
        endif()
    endforeach()
endif()
