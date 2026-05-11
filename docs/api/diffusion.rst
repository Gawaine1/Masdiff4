DiffusionModel 接口
===================

定义位置：``src/diffusion/base.py``

职责
----

``DiffusionModel`` 是条件扩散模型的抽象基类，负责：

- 随机初始化模型；
- 根据 ``Tau`` 生成奖励 ``R``（完整去噪采样）；
- 用种群中的 ``(Tau, R)`` 对训练模型；
- 对精英奖励做截断扩散变异（局部探索）。

抽象方法
--------

init_random
~~~~~~~~~~~

.. code-block:: python

   def init_random(self) -> None:
       ...

随机初始化模型参数。在主流程步骤 2 中调用。

generate_reward
~~~~~~~~~~~~~~~

.. code-block:: python

   def generate_reward(self, tau: Any) -> Any:
       ...

以 ``Tau`` 为条件，执行完整去噪过程生成奖励：

- **输入** ``tau``：环境仿真输出的条件向量；
- **输出** ``rewards``：生成的奖励矩阵，用于填充经验库并训练 DQN。

train_on_population
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def train_on_population(self, population: Population) -> None:
       ...

用当前种群训练扩散模型：

- **输入** ``population``：当前种群（``Individual`` 列表）；
- 实现中通常提取每个个体的 ``(tau, rewards)`` 作为训练样本对。

generate_reward_truncated
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def generate_reward_truncated(
       self,
       tau: Any,
       base_rewards: Any,
       *,
       add_noise_steps: int,
       denoise_steps: int,
   ) -> Any:
       ...

截断扩散变异：在精英奖励的邻域内生成新奖励：

- **输入** ``tau``：精英个体的条件向量；
- **输入** ``base_rewards``：精英个体原有的奖励 ``R``（变异起点）；
- **输入** ``add_noise_steps``：加噪步数，控制扰动幅度；
- **输入** ``denoise_steps``：去噪步数；
- **输出** ``mutated_rewards``：变异后的奖励。

.. tip::

   截断扩散的语义是：从精英奖励出发，先加少量噪声，再只做少量去噪，
   得到邻域内的新奖励样本。这是实现"**围绕优秀个体做局部探索**"的关键机制。
   ``add_noise_steps`` 和 ``denoise_steps`` 通过配置文件设置，越大变异幅度越大。

YAML 中声明
-----------

.. code-block:: yaml

   diffusion_model:
     class_path: "your_pkg.diffusion:YourDiffusionModel"
     kwargs:
       num_diffusion_steps: 1000
       beta_schedule: "cosine"

内置实现
--------

- ``src/diffusion/sumo_ryl_diffusion.py`` → ``SumoRylDiffusionModel``

  内部组件：

  - ``TauDiffusionModel``：噪声预测网络（条件 UNet 类型）；
  - ``NoiseScheduler``：DDPM 调度器；
  - ``DDIMScheduler``：DDIM 调度器（加速采样）。

  支持：

  - ``linear / cosine`` beta schedule；
  - ``ddpm / ddim`` 采样模式；
  - 输入 ``tau`` 形状：``[num_car, num_road, 2]``；
  - 输出 ``rewards`` 形状：``[num_car, num_road]``。

  详见 :doc:`../guide/sumo_ryl`。
