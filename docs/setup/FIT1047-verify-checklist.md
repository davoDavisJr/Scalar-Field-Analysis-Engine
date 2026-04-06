# FIT1047 Environment Capability Check

This document helps you confirm whether your computer and workspace are ready to complete FIT1047 tasks.

It does **not** install or configure anything.

If something fails, return to your unit’s official setup instructions.

If your environment is already working, return to the [main README](../../README.md).

---

## Expected capabilities

For FIT1047, your environment should allow you to:

* Edit MARIE assembly files (`.mas` / `.mar`) in VSCode (not required by course)
* Load, assemble, and run MARIE programs using the simulator (e.g. MARIE.js)
* Open and work with Logisim Evolution for basic digital circuit tasks

---

## Quick verification

### 1. MARIE workflow (primary)

1. Create a new file in VSCode:

   ```
   test.mas
   ```

2. Add a minimal program:

   ```
   / Simple test
   Load X
   Output
   Halt

   X, DEC 5
   ```

3. Save the file.

4. Open the MARIE simulator used in your unit ([https://marie.js.org/](https://marie.js.org/)).

5. Load or paste your program.

6. Assemble and run.

Expected result:

* Program assembles without errors
* Output is displayed correctly

If this fails:

* Do not attempt to fix this via the template repository
* Return to official FIT1047 setup instructions or lab guidance

---

### 2. VSCode editing check

* File opens without issues
* Text is readable and editable
* (Optional) Syntax highlighting is active if extension installed

---

### 3. Logisim Evolution (secondary)

1. Open Logisim Evolution
2. Create a new circuit
3. Place a basic component (e.g. input/output pin)

Expected result:

* Application launches
* Components can be placed and interacted with

If this fails:

* Follow official FIT1047 setup instructions

---

## Notes

* MARIE programming is the primary programming activity in FIT1047
* Logisim Evolution is used for selected tasks but is not the main workflow

---

## Summary

If all checks pass:

→ Your environment is ready for FIT1047 work

If any check fails:

→ Use official course setup instructions, not this repository
