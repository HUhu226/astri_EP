from math import log
import torch
import pickle
import numpy as np
import os
import hydra
from omegaconf import DictConfig
import logging
import treelib
from tensorboardX import SummaryWriter
import random
from src import Agent, Environment, ReplayBuffer
import matplotlib.pyplot as plt

class SolutionPool:
    def __init__(self, pool_size: int, alpha: float) -> None:
        self.pool_size = pool_size
        self.alpha = alpha

        self.best_hpwl = 1e9
        self.best_hpwl_max = 1e9

        self.sols = []
        self.tree = treelib.Tree()
        self.tree.create_node(
            tag="root",
            identifier=str([]),
            data={
                "visit_time": 0,
                "hpwl": 1e9,
                "hist_hpwl": {},
                "score": 0.0
            }
        )
        self.frontiers = [[]]
        self.frontiers_hist = [[]]

    @property
    def best_solution(self):
        return self.sols[0]["placement"], self.sols[0]["solution"], self.sols[0]["hpwl"]
    
    @property
    def depth_min_max(self):
        depth = [len(x) for x in self.frontiers]
        return min(depth), max(depth)
    
    def update_sol(self, frontier_id, solution, hpwl, placement):
        if hpwl < self.best_hpwl_max:
            sol_record = {
                "solution": solution,
                "placement": placement,
                "hpwl": hpwl,
            }
            self.sols.append(sol_record)
            self.sols = sorted(self.sols, key=lambda x: x["hpwl"])
            self.sols = self.sols[:self.pool_size]

            self.best_hpwl_max = self.sols[-1]["hpwl"]
            self.best_hpwl = self.sols[0]["hpwl"]

        for i in range(len(solution)):
            node = self.tree.get_node(str(solution[:i]))
            if node:
                node.data["visit_time"] += 1
                node.data["score"] = - node.data["hpwl"] + self.alpha * 1 / np.sqrt(node.data["visit_time"])
                
                if i < len(solution) - 1 and solution[:i+1] not in self.frontiers_hist:
                    node.data["hist_hpwl"][str(solution[i])] = hpwl.item() if str(
                        solution[i]) not in node.data["hist_hpwl"] else min(node.data["hist_hpwl"][str(solution[i])], hpwl.item())
                    node.data["hpwl"] = min(node.data["hpwl"], hpwl.item())
                    
            if node is None:
                self.tree.create_node(
                    tag=str(solution[i-1]),
                    identifier=str(solution[:i]),
                    parent=str(solution[:i-1]),
                    data={
                        "visit_time": 1,
                        "hpwl": hpwl.item(),
                        "score": - hpwl.item() + self.alpha * 1 / np.sqrt(1),
                        "hist_hpwl": {str(solution[i]): hpwl.item()},
                    }
                )

    def update_frontiers(self):
        front_scores = [(self.tree.get_node(str(x)).data["score"], x)
                        for x in self.frontiers]
        front_scores = sorted(
            front_scores, key=lambda x: self.tree.get_node(str(x[1])).data["hpwl"])
        assert len(front_scores) <= self.pool_size

        self.frontiers_cand = []
        for front in self.frontiers:
            for chid in self.tree.children(str(front)):
                if eval(chid.identifier) not in self.frontiers and eval(chid.identifier) not in self.frontiers_cand:
                    self.frontiers_cand.append(eval(chid.identifier))
        cand_scores = [(self.tree.get_node(str(x)).data["score"], x)
                       for x in self.frontiers_cand]
        cand_scores = sorted(
            cand_scores, key=lambda x: self.tree.get_node(str(x[1])).data["hpwl"])
        while len(cand_scores) > 0 and (cand_scores[0][0] >= front_scores[-1][0] or len(front_scores) < self.pool_size):
            node = self.tree.get_node(str(cand_scores[0][1]))
            parent = self.tree.parent(node.identifier)
            if node.tag in parent.data["hist_hpwl"]:
                parent.data["hist_hpwl"].pop(node.tag)
            parent.data["hpwl"] = 1e9 if len(parent.data["hist_hpwl"].values(
            )) == 0 else min(parent.data["hist_hpwl"].values())
            parent.data["score"] = - parent.data["hpwl"] + \
                self.alpha * 1 / np.sqrt(parent.data["visit_time"])

            front_scores.append(cand_scores[0])
            front_scores = sorted(
                front_scores, key=lambda x: x[0], reverse=True)
            self.frontiers.append(cand_scores[0][1])
            self.frontiers_hist.append(cand_scores[0][1])
            if len(self.frontiers) > self.pool_size:
                self.frontiers.remove(front_scores[-1][1])
                front_scores = front_scores[:-1]
            self.frontiers_cand.remove(cand_scores[0][1])
            cand_scores = cand_scores[1:]

    def sample(self):
        frontier_id = np.random.choice(len(self.frontiers))
        return frontier_id, self.frontiers[frontier_id]


class Trainer:

    def __init__(self, num_loops, num_episodes_in_loop, num_update_epochs, update_batch_size,
        num_macros_to_place, solution_pool_size, alpha, update_frontiers_begin, update_frontiers_freq,
        model_dir, solution_dir
    ):
        self.visualize_steps = [4, 8, 16, 32]  # 指定需要可视化的步骤
        self.step_positions = {step: [] for step in self.visualize_steps}  # 存储各步骤的位置数据
        self.env_grid = None
        self.visualize_freq = 100

        self.num_loops = num_loops
        self.num_episodes_in_loop = num_episodes_in_loop

        self.num_update_epochs = num_update_epochs
        self.update_batch_size = update_batch_size
        self.num_macros_to_place = num_macros_to_place

        self.buffer_capacity = num_episodes_in_loop * num_macros_to_place
        self.solution_pool_size = solution_pool_size
        self.alpha = alpha

        self.update_frontiers_begin = update_frontiers_begin
        self.update_frontiers_freq = update_frontiers_freq
        self.solution_pool = SolutionPool(solution_pool_size, alpha)

        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.solution_dir = solution_dir

    def setup_tb_writer(self, tb_writer: SummaryWriter):
        self.tb_writer = tb_writer

    def train(self, agent: Agent, env: Environment, save_dir="results"):

        logging.info(f"==================== Training ====================")
        self.env_grid = env.grid  # 在train方法中保存env.grid

        global_episode = 0

        for loop in range(self.num_loops):
            logging.info(
                f"-------------------- Loop {loop} --------------------")
            replay_buffer = ReplayBuffer(self.buffer_capacity, env.grid)
            for episode in range(self.num_episodes_in_loop):
                hpwl, sol_id, solution, placement = self.run_episode(
                    env, agent, replay_buffer, global_episode, save_dir)

                logging.info(f"Episode {global_episode}, HPWL = {hpwl}")
                self.solution_pool.update_sol(
                    sol_id, solution, hpwl, placement)

                self.tb_writer.add_scalar(
                    'episodes/hpwl', hpwl, global_episode)
                self.tb_writer.add_scalar(
                    "episodes/best_hpwl", self.solution_pool.best_hpwl, global_episode)
                self.tb_writer.add_scalar(
                    "episodes/best_hpwl_max", self.solution_pool.best_hpwl_max, global_episode)
                self.tb_writer.add_scalar(
                    "episodes/depth_min", self.solution_pool.depth_min_max[0], global_episode)
                self.tb_writer.add_scalar(
                    "episodes/depth_max", self.solution_pool.depth_min_max[1], global_episode)
                global_episode += 1

            agent.update(replay_buffer, self.num_update_epochs, self.update_batch_size)

            if global_episode >= self.update_frontiers_begin and loop % self.update_frontiers_freq == 0:
                self.solution_pool.update_frontiers()
            
            if global_episode % self.visualize_freq == 0:
                print(f"[INFO] Generating visualization at episode {global_episode}")
                self.visualize_step_positions(save_dir)

            placement, sol, hpwl = self.solution_pool.best_solution

            logging.info(f"Current best HPWL = {hpwl}.")
            torch.save(agent.actor_net.state_dict(),
                       os.path.join(self.model_dir, "actor_net.pth"))
            torch.save(agent.critic_net.state_dict(),
                       os.path.join(self.model_dir, "critic_net.pth"))

            os.makedirs(self.solution_dir, exist_ok=True)
            with open(os.path.join(self.solution_dir, f"best_placement_{hpwl}.pkl"), 'wb') as f:
                pickle.dump(placement, f)

                
        self.visualize_step_positions(save_dir)

    def run_episode(self, env: Environment, agent: Agent, replay_buffer: ReplayBuffer, global_episode: int, save_dir: str):
        # initialize the environment
        t, hpwl, score = 0, 0, 0
        solution, placement = [], []
        s = env.reset()
        done = False

        # randomly select a frontier to start an episode from
        sol_id, sol = self.solution_pool.sample()

        while True:
            if t < len(sol):
                # select the action from the solution
                a, a_logp = sol[t], 0.0
            else:
                a, a_logp = agent.select_action(s, t)

            s_, r, done, info = env.step(a, save_dir=save_dir, step=t, epoch=global_episode)
            placement.append(a)
            hpwl += info["delta_hpwl"]

            # 新增：记录指定步骤的位置（x, y）
            if t in self.visualize_steps:
                grid = env.grid
                pos_x = a // grid  # 计算x坐标（action是网格索引）
                pos_y = a % grid   # 计算y坐标
                self.step_positions[t].append((global_episode, pos_x, pos_y))  # 保存（epoch, x, y）
            
            if not done:
                if t >= len(sol):
                    replay_buffer.store(s, t, a, a_logp, r, s_, t + 1, 0.0)
                solution.append(a)
                score += r

                t += 1
            if done and env.num_macros > self.num_macros_to_place:
                # if the episode is done, place the remaining macros greedily
                s_done, s__done, a_done = s, s_, a
                s = s_
                for i in range(env.num_macros - self.num_macros_to_place):
                    a = agent.act_greedy(s)
                    s_, r_, done, info = env.step(a)
                    placement.append(a)
                    hpwl += info["delta_hpwl"]
                    r += r_
                    if i == env.num_macros - self.num_macros_to_place - 1:
                        replay_buffer.store(s_done, t, a_done, a_logp, r, s__done, t + 1, 1.0)
                        solution.append(a_done)
                    s = s_
                score += r
                break
                
            s = s_
            

        return hpwl * env.ratio / 1e5, sol_id, solution, placement
        
    def visualize_step_positions(self, save_dir):
        """为每个step生成一张图，包含所有episode的位置"""
        os.makedirs(f"{save_dir}/placement_visualization", exist_ok=True)
        
        for step in self.visualize_steps:
            positions = self.step_positions[step]
            if not positions:
                print(f"[WARNING] No positions recorded for step {step}")
                continue
            
            episodes, xs, ys = zip(*positions)
            
            plt.figure(figsize=(12, 10))
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.xlim(0, self.env_grid)
            plt.ylim(0, self.env_grid)
            
            # 使用颜色映射表示episode顺序（越新的episode颜色越深）
            colors = plt.cm.viridis(np.linspace(0, 1, max(episodes) + 1))
            
            # 为每个episode绘制位置点
            for ep, x, y in positions:
                plt.scatter(x, y, s=100, color=colors[ep], alpha=0.7, 
                            edgecolors='w', linewidths=1,
                            label=f'Episode {ep}' if ep % 50 == 0 else "")  # 每50个episode显示一次图例
            
            plt.title(f'Step {step} Placement Positions Across {len(positions)} Episodes')
            plt.xlabel('Grid X')
            plt.ylabel('Grid Y')
            
            # 添加颜色条表示episode
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=max(episodes)))
            sm.set_array([])
            cbar = plt.colorbar(sm, label='Episode')
            cbar.set_label('Episode')
            
            # 限制图例数量，避免过多
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper right', ncol=2, fontsize=8)
            
            # 保存图像（文件名包含当前最大episode数，方便查看随时间的变化）
            filename = f"step_{step}_all_episodes_up_to_{max(episodes)}.png"
            plt.savefig(f"{save_dir}/placement_visualization/{filename}", dpi=300, bbox_inches='tight')
            plt.close()


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(config: DictConfig):
    # set random seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    
    # set cpu num
    cpu_num = 10
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    
    # set device
    if(torch.cuda.is_available()): 
        device = torch.device(f'cuda:{config.cuda}')
        torch.cuda.empty_cache()
        logging.info(f'Using GPU: {torch.cuda.get_device_name()}, cuda: {config.cuda}.')
    else:
        device = torch.device('cpu')
        logging.info('Using CPU.')
    
    # set up tensorboard writer
    tb_writer = SummaryWriter(config.tb_dir)

    # instantiate agent, environment and trainer
    agent: Agent = hydra.utils.instantiate(config.agent)
    agent.set_up(device, tb_writer)
    env: Environment = hydra.utils.instantiate(config.env)
    trainer: Trainer = hydra.utils.instantiate(config.trainer)
    trainer.setup_tb_writer(tb_writer)

    # train the model
    trainer.train(agent, env, save_dir=config.env.save_dir)

if __name__ == '__main__':
    main()
