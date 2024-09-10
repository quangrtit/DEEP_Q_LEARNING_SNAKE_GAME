from src import Snake, Agent

def train_model(epochs=200):
    environment = Snake()
    agent = Agent(0.001, 1.0, 0.95, 0.1, 0.99, 64, 1000000)
    best_score = 0
    for i in range(epochs):
        state = environment.reset()
        time_step = 0
        while time_step < 10000:
            # action = environment.act()
            action = agent.choose_action(state)
            next_state, reward, done, score = environment.step(action)
            agent.save_experience(state, action, reward, next_state, done)
            environment.render()
            state = next_state
            # print("you use action", environment.covert_action[action])
            time_step += 1
            if done: 
                agent.update_target_NN()
                if score > best_score: 
                    best_score = score
                    agent.save_model("bach_duong_best.pth")
                print("epochs:", i, "score:", score - 1)
                break
            agent.train_one_bacth()
        agent.epsilon = min(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        if i > 1000: 
            agent.epsilon_min = 0.0
    agent.save_model("anh_quang.pth")
    environment.close_game()
if __name__ == "__main__":
    train_model()