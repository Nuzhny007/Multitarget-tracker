#pragma once

///
/// \brief MT_API: decoration for the public API.
///
/// Windows shared builds currently rely on CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS, so
/// MT_API is empty there; GCC/Clang get the default-visibility attribute, which
/// is a no-op until -fvisibility=hidden is enabled. This is the single place to
/// switch to explicit exports (dllexport/dllimport) later.
///
#if defined(_WIN32)
#  define MT_API
#elif defined(__GNUC__) || defined(__clang__)
#  define MT_API __attribute__((visibility("default")))
#else
#  define MT_API
#endif
