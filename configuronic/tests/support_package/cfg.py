import configuronic as cfgc
from configuronic.tests.support_package.subpkg.a import A
from configuronic.tests.support_package.b import B

a_cfg_value1 = cfgc.Config(A, value=1)
a_cfg_value2 = cfgc.Config(A, value=2)
b_cfg_value1 = cfgc.Config(B, value=1)
b_cfg_value2 = cfgc.Config(B, value=2)
