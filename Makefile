EXE ?= t
CC ?= null
# Get the absolute path of the Makefile
mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))

# Remove the /Makefile part
cleaned_path := $(subst /Makefile,/, $(mkfile_path))

# Construct the path to the executable
first_path := $(cleaned_path)dist/$(EXE).exe
second_path := $(cleaned_path)$(EXE).exe

first_path_windows := $(subst /,\,$(first_path))
second_path_windows := $(subst /,\,$(second_path))



all:
	pyinstaller --onefile --name $(EXE) engine.py consts.py eval.py moveorder.py tt.py timemanager.py
	cmd /c move $(first_path_windows) $(second_path_windows)
