import gym
import tensorflow as tf

USING_RANDOM = True

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='linear')
])

env = gym.make('CartPole-v1',render_mode='human')



for i_episode in range(2):
    score = 0
    observation = env.reset()[0]
    for t in range(100):
        env.render()
        print(observation)
        
        if USING_RANDOM:
            action = env.action_space.sample()
        else:
            action = model.predict(observation[None,:])
            action = tf.math.argmax(action, axis=1)#取得最大值的index
            action = action.numpy()[0]#轉成numpy array

        print(action)# 0 or 1 , 0: left , 1: right
        observation, reward, done,_ ,info = env.step(action)

        score += reward

        if done:
            print("Episode finished after {} timesteps , score: {}".format(t+1,score))
            break