from src import Snake, Agent
import numpy as np
def train_model(epochs=300):
    environment = Snake()
    environment.change_fps(30)
    agent = Agent(0.001, 1.0, 1.0/100, 0.0, 0.99, 64, 1000000)
    best_score = 0
    avg_score = 0
    score_history = []
    for i in range(epochs):
        state = environment.reset()
        time_step = 0
        while time_step < 50000:
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
                if score > best_score and score >= 60: 
                    best_score = score
                    agent.save_model("./src/models/model_" + str(best_score) + ".pth")
                avg_score = np.mean(score_history[-50:])
                print("epochs:", i, "score:", score - 1, "avg_score:", avg_score, "best_score:", best_score,  "epsilon:", agent.epsilon)
                break
            agent.train_one_bacth()
        agent.epsilon = max(agent.epsilon_min, agent.epsilon - agent.epsilon_decay)
        score_history.append(score)
        # if i > 1000: 
        #     agent.epsilon_min = 0.0
    agent.save_model("./src/models/last_model.pth")
    environment.close_game()
if __name__ == "__main__":
    train_model()