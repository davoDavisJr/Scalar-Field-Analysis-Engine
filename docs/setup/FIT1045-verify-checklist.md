# FIT1045 Environment Capability Check

This document helps you confirm whether your computer is **already correctly set up** for FIT1045.

It does **not** replace the official setup instructions.

If your environment is already working, return to the [main README](../../README.md).

---

## Official Setup Guide (Required Source of Truth)

If you have not yet set up your environment, follow the official FIT1045 guide:

**The Programmer’s Field Guide; Installation (Windows / MSYS2 / WSL)**
* Landing Page: <https://programmers.guide/>
* Installation Help: <https://programmers.guide/book/appendix/0-manual-installation/0-overview/>
* MSYS2: <https://programmers.guide/book/appendix/0-manual-installation/2-5-setup-win-msys/>
* WSL: <https://programmers.guide/book/appendix/0-manual-installation/2-6-setup-win-wsl/>

This template assumes that you have already completed that setup.

---

## What your computer is expected to support

After completing the official setup, your machine should be able to:

### Core capabilities

* Compile C/C++ programs using `clang++` or `g++`
* Run compiled executables from a terminal
* Use `git` for version control
* Open and edit code in VSCode

### SplashKit capabilities

* Compile programs that include:

  ```cpp
  #include "splashkit.h"
  ```
* Link against the SplashKit library
* Run simple graphical or interactive programs

### Terminal environment

* Use a terminal environment provided by:

  * MSYS2 (MINGW64), or
  * WSL
* Run compilation commands inside that environment

---

## Quick verification (recommended)

Open your terminal (MSYS2 MINGW64 or WSL) and test the following.

### 1. Compiler check

```bash
clang++ --version
```

or

```bash
g++ --version
```

If this prints version information, your compiler is available.

---

### 2. Basic compilation test

Create a file called `test.cpp`:

```cpp
#include <iostream>
int main() {
    std::cout << "Hello, world!" << std::endl;
}
```

Compile and run:

```bash
clang++ test.cpp -o test
./test
```

You should see:

```text
Hello, world!
```

---

### 3. SplashKit verification (course-aligned)

FIT1045 uses SplashKit.

The Field Guide’s intended setup uses a global SplashKit installation so the compiler can find the required libraries and headers automatically.

Run:

```bash
skm update
skm global install
```

(or compile a SplashKit example provided in the Field Guide)

If this succeeds, SplashKit is correctly installed.

---

## If something does not work

Do not attempt to fix the issue using this repository.

Instead:

1. Return to the official setup guide
2. Re-check each step carefully
3. Ensure you are using the correct terminal environment (MSYS2 MINGW64 or WSL)

---

## Summary

If all checks pass, your environment is considered **FIT1045-compliant** and you can proceed to use this template.

If not, follow the official setup instructions before continuing.

(Return to the [main README](../../README.md))