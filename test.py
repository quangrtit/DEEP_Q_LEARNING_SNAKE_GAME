from src import Snake, Agent

def agent_play():
    environment = Snake()
    environment.change_fps(60)
    agent = Agent(0.001, 0.0, 0.0, 0.0, 0.99, 64, 1000000)
    agent.load_model("./src/models/bach_duong_best.pth")
    end_game = False
    while not end_game:
        state = environment.reset()
        while True:
            # action = environment.act()
            action = agent.choose_action(state)
            next_state, reward, done, score = environment.step(action)
            environment.render()
            state = next_state
            # print("you use action", environment.covert_action[action])
            if done: 
                break
        environment.delay_game()
if __name__ == "__main__":
    agent_play()