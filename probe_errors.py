from probe_run import Policy, Reset, Robot

policy, robot = Policy(), Robot()

policy.reset(Reset(home=True))          # ok
policy.reset(home=True)                 # not the message type
policy.reset(Reset(home='yes'))         # wrong field type

for call in robot.reset.incoming():
    call.set_result(True)               # ok
    call.set_result('done')             # Res is bool
    call.request.hom                    # typo in a field name
