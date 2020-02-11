from atari_norm import *

dqn = DQN()
target = DQN()
target.cuda()
print(dqn.cuda())

scores = train(dqn, target, env, test_set=get_test_set(), replay_buf=RingBuffer(100000), recent=False, max_epochs=40e5)
