import os
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(os.path.abspath("app.py"), default_timeout=180).run()
assert not at.exception, at.exception
for p in ["Predict", "Explore properties", "Market insights", "Model details", "Methodology & limitations"]:
    at.sidebar.radio[0].set_value(p).run()
    assert not at.exception, (p, at.exception)
    print("OK ", p)
print("SMOKE TEST PASSED")