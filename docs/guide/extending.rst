扩展指南
========

MASDiff 的核心价值之一是便于扩展。只要实现了约定接口并在 YAML 中声明，即可替换任意模块。

扩展新环境
----------

继承 ``src/environments/base.py`` 中的 ``Environment``，实现两个方法：

.. code-block:: python

   from src.environments.base import Environment

   class MyEnvironment(Environment):

       def simulate_collect(self, policies):
           # 运行仿真，返回 (experience_buffers, tau)
           ...

       def simulate_evaluate(self, policies):
           # 运行仿真，返回 simulation_data（与 Q 格式一致）
           ...

在 YAML 中声明：

.. code-block:: yaml

   environment:
     class_path: "my_pkg.envs:MyEnvironment"
     kwargs:
       map_file: "maps/my_map.net.xml"

扩展新 DQN 模块
---------------

继承 ``src/dqn/base.py`` 中的 ``DqnModule``：

.. code-block:: python

   from src.dqn.base import DqnModule

   class MyDqnModule(DqnModule):

       def init_random_policies(self, num_agents):
           # 返回长度为 num_agents 的随机策略列表
           ...

       def build_training_data(self, experience_buffers, rewards):
           # 用 rewards 填充 experience_buffers 中的 r，返回训练数据
           ...

       def train_per_agent(self, training_data):
           # 为每个 agent 训练，返回策略列表
           ...

扩展新扩散模型
--------------

继承 ``src/diffusion/base.py`` 中的 ``DiffusionModel``：

.. code-block:: python

   from src.diffusion.base import DiffusionModel

   class MyDiffusionModel(DiffusionModel):

       def init_random(self):
           ...

       def generate_reward(self, tau):
           # 完整去噪生成奖励
           ...

       def train_on_population(self, population):
           # 从 population 中提取 (tau, rewards) 对训练模型
           ...

       def generate_reward_truncated(self, tau, base_rewards, *, add_noise_steps, denoise_steps):
           # 截断扩散：加噪 add_noise_steps 步，去噪 denoise_steps 步
           ...

扩展新评价指标
--------------

继承 ``src/metrics/base.py`` 中的 ``Metric``：

.. code-block:: python

   from src.metrics.base import Metric

   class MyMetric(Metric):

       def compute_rho(self, q, simulation_data) -> float:
           # 计算 simulation_data 相对于 q 的适应度（越大越好）
           ...

扩展精英选择策略
----------------

继承 ``src/evolution/base.py`` 中的 ``EliteSelector``：

.. code-block:: python

   from src.evolution.base import EliteSelector

   class MyEliteSelector(EliteSelector):

       def select_elites(self, population, *, elite_count):
           # 从 population 中选出 elite_count 个精英个体
           ...

扩展并行执行器
--------------

继承 ``src/parallel/base.py`` 中的 ``ParallelExecutor``：

.. code-block:: python

   from src.parallel.base import ParallelExecutor

   class MyExecutor(ParallelExecutor):

       def map(self, fn, items):
           # 并行或串行执行 fn(item) for item in items
           ...

       def close(self):
           # 释放资源
           ...

数据契约注意事项
----------------

各模块之间存在隐式的数据格式约定，框架本身不做格式校验：

.. list-table::
   :widths: 30 30 40
   :header-rows: 1

   * - 数据
     - 生产者
     - 消费者
   * - ``tau``
     - ``Environment.simulate_collect``
     - ``DiffusionModel.generate_reward``
   * - ``rewards``
     - ``DiffusionModel.generate_reward``
     - ``DqnModule.build_training_data``
   * - ``experience_buffers``
     - ``Environment.simulate_collect``
     - ``DqnModule.build_training_data``
   * - ``simulation_data``
     - ``Environment.simulate_evaluate``
     - ``Metric.compute_rho``
   * - ``q``
     - ``QProvider.load_or_create_q``
     - ``Metric.compute_rho``

扩展时需保证同一实验中，这些数据在生产者和消费者之间的格式完全一致。
