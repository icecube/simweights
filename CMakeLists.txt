i3_project(simweights
  PYTHON_DIR python
  )

i3_test_scripts(
  resources/tests/compare_fluxes.py
  resources/tests/test_fluxes.py
  )
