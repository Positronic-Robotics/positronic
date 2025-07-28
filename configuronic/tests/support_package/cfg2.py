import configuronic as cfn
from configuronic.tests.support_package.cfg import a_cfg_value1, b_cfg_value1

a_cfg_value1_copy = a_cfg_value1.copy()
a_cfg_value1_override_value3 = a_cfg_value1.override(value=3)
a_nested_b_value1 = a_cfg_value1.override(value=b_cfg_value1)


@cfn.config()
def return2():
    return 2
