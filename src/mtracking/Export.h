#pragma once

///
/// \brief MT_API: decoration for the public API.
///
/// When the project is built as shared libraries (`MT_SHARED`), the library
/// defines `MT_SHARED_EXPORTS` and its public classes/functions are exported;
/// consumers get `MT_SHARED` and import them. Static builds define neither, so
/// `MT_API` is empty. GCC/Clang get the default-visibility attribute, which is
/// a no-op unless `-fvisibility=hidden` is enabled.
///
#if defined(MT_SHARED_EXPORTS)
#  if defined(_WIN32)
#    define MT_API __declspec(dllexport)
#  elif defined(__GNUC__) || defined(__clang__)
#    define MT_API __attribute__((visibility("default")))
#  else
#    define MT_API
#  endif
#elif defined(MT_SHARED)
#  if defined(_WIN32)
#    define MT_API __declspec(dllimport)
#  else
#    define MT_API
#  endif
#else
#  define MT_API
#endif
