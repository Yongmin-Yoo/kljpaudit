# Data schema

The experiments use the LJP-Criminal subset of LBOX OPEN.

The model input is restricted to the `facts` field. The ruling, sentencing
reason, case name, and textual judgment label are excluded from the input.

The three prediction targets are:

- `fine_lv`
- `imprisonment_with_labor_lv`
- `imprisonment_without_labor_lv`

The dataset is not redistributed through this repository. Users must obtain it
from the official LBOX OPEN source and comply with its license.
