pass_test=sample_maker.generate_narrowband_1()
pass_test=pass_test.flatten()
fail_test=get_sample(180,560)
fail_test=process_sample(fail_test)

pass_result=APF_nn.feedforward(pass_test)
fail_result=APF_nn.feedforward(fail_test)
